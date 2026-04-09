import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import deque


def smooth_raw_data_v2(variable_df, dx, dt, smooth_x_window=0.5, smooth_t_window=36, 
                       c_free=45, c_cong=-13, x_offset=0.0):
    """
    Optimized ASM (Adaptive Smoothing Method) implementation.
    
    Parameters:
    -----------
    variable_df : DataFrame with columns ['t_index', 'x_index', 'variable']
    dx : float - spatial resolution in miles
    dt : float - temporal resolution in minutes
    smooth_x_window : float - spatial smoothing window in miles
    smooth_t_window : float - temporal smoothing window in minutes
    c_free : float - free flow wave speed in mph
    c_cong : float - congestion wave speed in mph
    x_offset : float - spatial offset to add to x coordinates

    Returns:
    --------
    DataFrame with columns ['t', 'x', 'raw_speed', 'time', 'milemarker', 'speed']
    """
    
    # Create the output dataframe - handle if input is already a dataframe
    if isinstance(variable_df, pd.DataFrame):
        result = variable_df[['t_index', 'x_index', 'variable']].copy()
    else:
        result = pd.DataFrame(variable_df)
        result.columns = ['t_index', 'x_index', 'variable']
    
    result['t'] = dt * result['t_index']
    result['x'] = dx * result['x_index'] + x_offset
    
    # Extract only non-NaN data points for interpolation
    mask_data = result['variable'].notna()
    data_points = result[mask_data][['t', 'x', 'variable']].values
    
    if len(data_points) == 0:
        # No data to smooth
        result['EGTF'] = np.nan
        result.columns = ['t', 'x', 'raw_variable', 'time', 'milemarker', 'variable']
        return result
    
    # Get all points where we need to compute smoothed values
    query_points = result[['t', 'x']].values
    
    # Build a KD-tree for efficient spatial-temporal queries
    # Normalize coordinates to handle different units (time in minutes, space in miles)
    # Scale factors to make the window roughly isotropic
    t_scale = 1.0 / (smooth_t_window / 2)
    x_scale = 1.0 / (smooth_x_window / 2)
    
    data_coords_scaled = data_points[:, :2].copy()
    data_coords_scaled[:, 0] *= t_scale
    data_coords_scaled[:, 1] *= x_scale
    
    query_coords_scaled = query_points.copy()
    query_coords_scaled[:, 0] *= t_scale
    query_coords_scaled[:, 1] *= x_scale
    
    # Build KD-tree
    tree = cKDTree(data_coords_scaled)
    
    # Query for all neighbors within the window (using scaled distance of sqrt(2))
    # This captures all points within the rectangular window
    neighbors_list = tree.query_ball_point(query_coords_scaled, r=np.sqrt(2))
    
    # Vectorized computation of smoothed speeds
    smoothed_speeds = np.zeros(len(query_points))
    
    # Pre-compute constants
    c_free_factor = 3600.0 / c_free
    c_cong_factor = 3600.0 / c_cong
    
    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) == 0:
            smoothed_speeds[i] = 80.0  # Default value
            continue
        
        # Get query point coordinates
        t_q = query_points[i, 0]
        x_q = query_points[i, 1]
        
        # Get neighbor data
        t_neighbors = data_points[neighbors, 0]
        x_neighbors = data_points[neighbors, 1]
        speed_neighbors = data_points[neighbors, 2]
        
        # Compute dx and dt vectors
        dx_vec = x_q - x_neighbors
        dt_vec = t_q - t_neighbors
        
        # Filter neighbors within the rectangular window
        mask = (np.abs(dt_vec) <= smooth_t_window / 2) & (np.abs(dx_vec) <= smooth_x_window / 2)
        
        if not np.any(mask):
            smoothed_speeds[i] = 80.0  # Default value
            continue
        
        # Apply mask
        dx_vec = dx_vec[mask]
        dt_vec = dt_vec[mask]
        speed_neighbors = speed_neighbors[mask]
        
        # Compute beta weights for free flow
        dt_free = dt_vec - c_free_factor * dx_vec
        beta_free = np.exp(-(np.abs(dx_vec) / smooth_x_window + np.abs(dt_free) / smooth_t_window))
        
        # Compute beta weights for congested flow
        dt_cong = dt_vec - c_cong_factor * dx_vec
        beta_cong = np.exp(-(np.abs(dx_vec) / smooth_x_window + np.abs(dt_cong) / smooth_t_window))
        
        # Compute weighted averages
        sum_beta_free = np.sum(beta_free)
        sum_beta_cong = np.sum(beta_cong)
        
        if sum_beta_free > 0 and sum_beta_cong > 0:
            EGTF_v_free = np.sum(beta_free * speed_neighbors) / sum_beta_free
            EGTF_v_cong = np.sum(beta_cong * speed_neighbors) / sum_beta_cong
        else:
            EGTF_v_free = 80.0
            EGTF_v_cong = 80.0
        
        # Compute adaptive weight
        v = min(EGTF_v_free, EGTF_v_cong)
        tanh_term = np.tanh((40.0 - v) / 10)
        w = 0.5 * (1.0 + tanh_term)
        
        # Compute final smoothed speed
        smoothed_speeds[i] = w * EGTF_v_cong + (1.0 - w) * EGTF_v_free
    
    result['EGTF'] = smoothed_speeds
    result.columns = ['t', 'x', 'raw_variable', 'time', 'milemarker', 'variable']
    
    return result

def summarize_sparsity(
    frame: pd.DataFrame,
    label: str,
    mode: str = "by_station",
    group_col: str = "id",
    id_col: Optional[str] = None,
    top_n: int = 25,
    figsize: Tuple[int, int] = (10, 4),
) -> pd.DataFrame:
    """
    Unified sparsity summary utility with plotting + clean tabular output.

    Modes:
    - by_station: null percentage by column within each group_col value.
    - column_nulls: null counts and percentages by column.
    - lane_all_null_by_id: rows where all lane columns are null, grouped by station ID.
    """
    if frame.empty:
        print(f"{label}: frame is empty.")
        return pd.DataFrame()

    def _finalize_plot(title: str, xlabel: str, ylabel: str) -> None:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    if mode == "by_station":
        if group_col not in frame.columns:
            raise ValueError(f"Group column '{group_col}' not found in frame.")

        cols_to_score = [c for c in frame.columns if c != group_col]
        if not cols_to_score:
            raise ValueError(
                f"No columns available to summarize after excluding group column '{group_col}'."
            )

        summary = (
            frame.groupby(group_col)[cols_to_score]
            .apply(lambda x: x.isnull().mean())
            .reset_index()
            .melt(id_vars=[group_col], var_name="column", value_name="percent_null")
            .sort_values([group_col, "percent_null"], ascending=[True, False])
            .reset_index(drop=True)
        )
        summary["percent_null"] = summary["percent_null"].round(3)

        print(f"{label} null summary (by {group_col}):")
        print(f"Total rows: {len(frame):,}")

        heatmap_df = summary.pivot(index=group_col, columns="column", values="percent_null")
        dynamic_height = max(figsize[1], 0.35 * len(heatmap_df.index) + 2)
        plt.figure(figsize=(figsize[0], dynamic_height))
        plt.imshow(heatmap_df.values, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        plt.colorbar(label="Percent Null")
        plt.xticks(np.arange(len(heatmap_df.columns)), heatmap_df.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(heatmap_df.index)), heatmap_df.index.astype(str))
        _finalize_plot(
            title=f"{label}: Null Share by {group_col} and column",
            xlabel="Column",
            ylabel=group_col,
        )
        return summary

    if mode == "column_nulls":
        null_count = frame.isnull().sum()
        summary = pd.DataFrame(
            {
                "column": null_count.index,
                "null_count": null_count.values.astype(int),
            }
        )
        summary["percent_null"] = (summary["null_count"] / len(frame)).round(3)
        summary = summary[summary["null_count"] > 0].sort_values(
            "null_count", ascending=False
        )

        print(f"{label} sensor readings: {len(frame):,}")

        if not summary.empty:
            plot_df = summary.head(top_n)
            plt.figure(figsize=figsize)
            plt.bar(plot_df["column"], plot_df["percent_null"])
            plt.xticks(rotation=45, ha="right")
            _finalize_plot(
                title=f"{label}: Null Share by Column (top {len(plot_df)})",
                xlabel="Column",
                ylabel="Percent Null",
            )
        return summary.reset_index(drop=True)

    if mode == "lane_all_null_by_id":
        lane_cols = [col for col in frame.columns if "lane" in col.lower()]
        if not lane_cols:
            print(f"{label}: no columns containing 'lane' found.")
            return pd.DataFrame()

        resolved_id_col = id_col or ("ID" if "ID" in frame.columns else "id")
        if resolved_id_col not in frame.columns:
            raise ValueError(
                f"ID column '{resolved_id_col}' not found. Pass id_col explicitly."
            )

        all_lane_null = frame[lane_cols].isna().all(axis=1)
        total_rows = frame.groupby(resolved_id_col).size().rename("total_rows")
        null_rows = (
            frame.loc[all_lane_null]
            .groupby(resolved_id_col)
            .size()
            .rename("all_lane_null_rows")
        )
        summary = (
            total_rows.to_frame()
            .join(null_rows, how="left")
            .fillna(0)
            .astype({"all_lane_null_rows": int})
            .reset_index()
        )
        summary["pct_all_lane_null"] = (
            summary["all_lane_null_rows"] / summary["total_rows"]
        ).round(3)
        summary = summary.sort_values("all_lane_null_rows", ascending=False)

        print(f"{label} lane-null summary (per {resolved_id_col}):")
        print(f"Total rows: {len(frame):,}")

        plot_df = summary.head(top_n)
        plt.figure(figsize=figsize)
        plt.bar(plot_df[resolved_id_col].astype(str), plot_df["pct_all_lane_null"])
        plt.xticks(rotation=45, ha="right")
        _finalize_plot(
            title=f"{label}: All-Lane-Null Share by {resolved_id_col} (top {len(plot_df)})",
            xlabel=resolved_id_col,
            ylabel="Percent All-Lane-Null",
        )
        return summary.reset_index(drop=True)

    raise ValueError(
        "Unsupported mode. Use one of: 'by_station', 'column_nulls', 'lane_all_null_by_id'."
    )

def increase_resolution(matrix: np.ndarray, space_factor, time_factor) -> np.ndarray:
    return np.repeat(matrix, space_factor, axis=0).repeat(time_factor, axis=1)

def subdivide_space_bins(space_bins, factor):
    """
    Split each interval in `space_bins` into `factor` equal parts.

    Example: space_bins=[83, 85] with factor=4 → [83, 83.5, 84, 84.5, 85]
    """
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    edges = np.asarray(space_bins, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("space_bins must be a 1D sequence with ≥2 edges")

    refined = [edges[0]]
    for start, end in zip(edges[:-1], edges[1:]):
        segment_edges = np.linspace(start, end, factor + 1)[1:]
        refined.extend(segment_edges.tolist())
    return np.array(refined)

# def df_to_matrix(variable_df, num_time_bins=None, num_space_bins=None, impute=True):
#     """Convert a long-form DataFrame back into a matrix shaped (time, space)."""
#     required = {"t", "x", "variable"}
#     if not required.issubset(variable_df.columns):
#         raise ValueError("DataFrame must contain t, x, variable columns")

#     if num_time_bins is None:
#         num_time_bins = int(variable_df["t"].max()) + 1
#     if num_space_bins is None:
#         num_space_bins = int(variable_df["x"].max()) + 1

#     matrix = np.full((num_time_bins, num_space_bins), np.nan)

#     t_idx = variable_df["t"].astype(int).to_numpy()
#     x_idx = variable_df["x"].astype(int).to_numpy()
    
#     if impute:
#         if "raw_variable" in variable_df.columns:
#             raw_vals = variable_df["raw_variable"].to_numpy()
#             smooth_vals = variable_df["variable"].to_numpy()
#             values = np.where(np.isnan(raw_vals), smooth_vals, raw_vals)
#         else:
#             values = variable_df["variable"].to_numpy()

#     else:
#         values = variable_df["variable"].to_numpy()

#     matrix[t_idx, x_idx] = values
#     return matrix

def df_to_matrix(df, time_column, space_column, value_column):
    """Convert long-form data to a matrix with shape (space, time)."""
    required = {time_column, space_column, value_column}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    t = df[time_column].to_numpy()
    s = df[space_column].to_numpy()

    # Enforce integer index columns
    if not np.all(np.equal(t, t.astype(int))):
        raise ValueError(f"{time_column} must contain integer indices")
    if not np.all(np.equal(s, s.astype(int))):
        raise ValueError(f"{space_column} must contain integer indices")

    t_idx = t.astype(int)
    s_idx = s.astype(int)
    values = df[value_column].to_numpy()

    num_time_bins = t_idx.max() + 1
    num_space_bins = s_idx.max() + 1

    # rows = space, columns = time
    matrix = np.full((num_space_bins, num_time_bins), np.nan)
    matrix[s_idx, t_idx] = values

    return matrix

def average_neighbors_y(matrix, num_neighbors=3):
    smoothed = matrix.astype(float, copy=True)
    weight_template = np.arange(num_neighbors + 1, 0, -1)
    full_weights = np.concatenate([weight_template[1:], weight_template])
    for i in range(num_neighbors, matrix.shape[0] - num_neighbors):
        for j in range(matrix.shape[1]):
            window = matrix[i - num_neighbors : i + num_neighbors + 1, j]
            valid_mask = ~np.isnan(window)
            if not valid_mask.any():
                smoothed[i, j] = np.nan
                continue

            smoothed[i, j] = np.average(
                window[valid_mask],
                weights=full_weights[valid_mask],
            )
    return smoothed

def process_pems(
    df,
    time_col,
    postmile_col,
    value_col,
    start_pm,
    end_pm,
    time_interval,
    space_interval,
    t_min = None,
    t_max = None
):
    """
    Bin values into a time-space matrix.

    Output orientation:
    - rows: space (bottom row corresponds to start_pm)
    - cols: time (left to right)
    """
    df = df.copy()

    # Validate inputs
    required_columns = [time_col, postmile_col, value_col]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            "Input DataFrame must contain the following columns: {required_columns}"
        )
    if start_pm == end_pm:
        raise ValueError("start_pm and end_pm must differ.")

    # Build time bins
    if t_min is not None and t_max is not None:
        time_bins = pd.date_range(start=t_min, end=t_max, freq=time_interval)
    else:
        time_bins = pd.date_range(
            start=df[time_col].min(),
            end=df[time_col].max(),
            freq=time_interval,
        )

    if len(time_bins) < 2:
        raise ValueError("Not enough time range for the requested time_interval.")

    # Build uniform space bins and use ascending edges for pd.cut
    travel_bins = _build_space_bins(start_pm, end_pm, space_interval)
    cut_bins = np.sort(travel_bins)
    num_space_bins = len(cut_bins) - 1

    # Assign bin indices
    df["time_bin"] = pd.cut(
        df[time_col], bins=time_bins, labels=False, include_lowest=True
    )
    df["space_bin_asc"] = pd.cut(
        df[postmile_col], bins=cut_bins, labels=False, include_lowest=True
    )

    # Drop out-of-range points and cast bin indices
    df = df.dropna(subset=["time_bin", "space_bin_asc"]).astype(
        {"time_bin": int, "space_bin_asc": int}
    )

    # Row mapping so TOP row corresponds to start_pm and BOTTOM row to end_pm
    if start_pm < end_pm:
        # Ascending PM: ascending bins already put start_pm at top (row 0)
        df["space_bin"] = df["space_bin_asc"].astype(int)
    else:
        # Descending PM: reverse so start_pm maps to top row
        df["space_bin"] = (num_space_bins - 1 - df["space_bin_asc"]).astype(int)

    # Build matrix with rows=space, cols=time
    num_time_bins = len(time_bins) - 1
    value_matrix = np.full((num_space_bins, num_time_bins), np.nan)
    grouped = df.groupby(["space_bin", "time_bin"])[value_col].mean()
    for (space_bin, time_bin), value in grouped.items():
        value_matrix[space_bin, time_bin] = value

    # Convert to df
    value_df = matrix_to_df(value_matrix)
    return value_matrix, value_df


def matrix_to_df(value_matrix: np.ndarray) -> pd.DataFrame:
    """
    Convert a value matrix to long format with integer indices.

    Returns columns: space_index, time_index, value
    """
    if value_matrix.ndim != 2:
        raise ValueError("value_matrix must be 2D.")

    num_space_bins, num_time_bins = value_matrix.shape
    space_idx, time_idx = np.meshgrid(
        np.arange(num_space_bins), np.arange(num_time_bins), indexing="ij"
    )
    return pd.DataFrame(
        {
            "space_index": space_idx.ravel(),
            "time_index": time_idx.ravel(),
            "value": value_matrix.ravel(),
        }
    )

# def plot_matrix(
#     matrix,
#     value_label,
#     t_min,
#     t_max,
#     start_pm,
#     end_pm):
#     """
#     Plot a single heatmap with inferred time/space increments.

#     If space_bins are provided, they define non-uniform spatial bin edges.
#     """
#     num_space_bins, num_time_bins = matrix.shape
#     time_ticks = np.linspace(0, num_time_bins - 1, min(10, num_time_bins)).astype(int)
#     space_ticks = np.linspace(0, num_space_bins - 1, min(10, num_space_bins)).astype(
#         int
#     )

#     if t_max <= t_min:
#         raise ValueError("t_max must be greater than t_min to infer time spacing.")
#     inferred_time_increment = (t_max - t_min) / num_time_bins

#     if start_pm == end_pm:
#         raise ValueError("start_pm must differ from end_pm to infer spacing.")
#     inferred_space_increment = (end_pm - start_pm) / num_space_bins
#     print('inferred space increment:', round(inferred_space_increment * 1.60934, 2), 'km')
#     space_edges = start_pm + np.arange(num_space_bins + 1) * inferred_space_increment

#     # Scale figure dimensions with actual axis ranges so halving a range halves the plot span
#     time_range_hours = max((t_max - t_min).total_seconds() / 3600.0, 1e-6)
#     space_extent_miles = abs(space_edges[-1] - space_edges[0])
#     space_range_miles = max(space_extent_miles, 1e-6)
#     width_per_hour = 5
#     height_per_mile = 1
#     fig_width = np.clip(time_range_hours * width_per_hour, 3.0, 20.0)
#     fig_height = np.clip(space_range_miles * height_per_mile, 3.0, 20.0)
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))

#     # Use index-based y positions so labels are evenly spaced on the axis.
#     pm_by_row = np.linspace(end_pm, start_pm, num_space_bins)
#     space_tick_values = space_ticks
#     space_labels = [round(float(pm_by_row[idx]), 2) for idx in space_ticks]

#     print('inferred time increment:', inferred_time_increment) 
#     print('num time bins:', num_time_bins)
#     print('num space bins:', num_space_bins)
#     print('num space edges:', len(space_edges) - 1)

#     pcm = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", interpolation="nearest")
#     cbar = fig.colorbar(pcm, ax=ax)
#     cbar.set_label(value_label)

#     time_tick_times = [t_min + int(i) * inferred_time_increment for i in time_ticks]
#     ax.set_xticks(time_ticks)
#     ax.set_xticklabels([ts.strftime("%H:%M") for ts in time_tick_times], rotation=45)
#     ax.set_yticks(space_tick_values)
#     ax.set_yticklabels(space_labels)
#     ax.set_ylabel("Space (Postmile Abs)")
#     ax.set_xlabel("Time")

#     time_res_sec = int(inferred_time_increment.total_seconds())
#     space_res_miles = abs(round(inferred_space_increment, 2))
#     title = (
#             f"Time-Space Diagram of {value_label}\n"
#             f"Resolution: {time_res_sec} sec x {space_res_miles} miles"
#         )
#     plt.title(title)
#     fig.tight_layout()
#     plt.show()


def plot_matrix(
    matrix,
    title,
    colorbar_label=None,
    t_min=None,
    t_max=None,
    start_pm=None,
    end_pm=None,
):
    """
    Plot a single heatmap with optional inferred time/space resolution.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix with shape (space, time).
    title : str
        Used in the figure title: "Time-Space Diagram for {title}".
    colorbar_label : str, optional
        Label for the colorbar. If omitted, uses `title`.
    t_min, t_max : optional
        If both are provided, label the x-axis with timestamps and infer temporal resolution.
        If neither is provided, use integer time indices.
    start_pm, end_pm : optional
        If both are provided, label the y-axis with rounded integer PMs and infer spatial resolution.
        If neither is provided, use integer space indices.
    """
    if (t_min is None) != (t_max is None):
        raise ValueError("Pass both t_min and t_max, or neither.")
    if (start_pm is None) != (end_pm is None):
        raise ValueError("Pass both start_pm and end_pm, or neither.")

    num_space_bins, num_time_bins = matrix.shape
    time_ticks = np.linspace(0, num_time_bins - 1, min(10, num_time_bins)).astype(int)
    space_ticks = np.linspace(0, num_space_bins - 1, min(10, num_space_bins)).astype(int)

    colorbar_label = colorbar_label or title

    inferred_time_increment = None
    inferred_space_increment = None
    space_edges = None

    if t_min is not None:
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min to infer time spacing.")
        inferred_time_increment = (t_max - t_min) / num_time_bins
        print("inferred time increment:", inferred_time_increment)

    if start_pm is not None:
        if start_pm == end_pm:
            raise ValueError("start_pm must differ from end_pm to infer spacing.")
        inferred_space_increment = (end_pm - start_pm) / num_space_bins
        print(
            "inferred space increment:",
            round(inferred_space_increment * 1.60934, 2),
            "km",
        )
        space_edges = start_pm + np.arange(num_space_bins + 1) * inferred_space_increment

    print("num time bins:", num_time_bins)
    print("num space bins:", num_space_bins)
    if space_edges is not None:
        print("num space edges:", len(space_edges) - 1)

    # Scale figure dimensions with actual axis ranges when available.
    if inferred_time_increment is not None:
        time_range_hours = max((t_max - t_min).total_seconds() / 3600.0, 1e-6)
    else:
        time_range_hours = max(num_time_bins / 12.0, 1e-6)

    if space_edges is not None:
        space_extent_miles = abs(space_edges[-1] - space_edges[0])
        space_range_miles = max(space_extent_miles, 1e-6)
    else:
        space_range_miles = max(num_space_bins / 10.0, 1e-6)

    width_per_hour = 5
    height_per_mile = 1
    fig_width = np.clip(time_range_hours * width_per_hour, 3.0, 20.0)
    fig_height = np.clip(space_range_miles * height_per_mile, 3.0, 20.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pcm = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", interpolation="nearest", origin="lower")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xticks(time_ticks)
    if inferred_time_increment is not None:
        time_tick_times = [t_min + int(i) * inferred_time_increment for i in time_ticks]
        ax.set_xticklabels([ts.strftime("%H:%M") for ts in time_tick_times], rotation=45)
        ax.set_xlabel("Time")
    else:
        ax.set_xticklabels(time_ticks)
        ax.set_xlabel("Time Index")

    ax.set_yticks(space_ticks)
    if inferred_space_increment is not None:
        # Use evenly spaced tick positions, but label them with rounded integer PM values.
        pm_by_row = np.linspace(start_pm, end_pm, num_space_bins)
        space_labels = [int(round(float(pm_by_row[idx]))) for idx in space_ticks]
        ax.set_yticklabels(space_labels)
        ax.set_ylabel("Space (Postmile Abs)")
    else:
        ax.set_yticklabels(space_ticks)
        ax.set_ylabel("Space Index")

    title_lines = [f"Time-Space Diagram for {title}"]
    resolution_parts = []

    if inferred_time_increment is not None:
        time_res_sec = int(inferred_time_increment.total_seconds())
        resolution_parts.append(f"{time_res_sec} sec")

    if inferred_space_increment is not None:
        space_res_miles = abs(round(inferred_space_increment, 2))
        resolution_parts.append(f"{space_res_miles} miles")

    if resolution_parts:
        title_lines.append("Resolution: " + " x ".join(resolution_parts))

    ax.set_title("\n".join(title_lines))
    fig.tight_layout()
    plt.show()

def _build_space_bins(x_start, x_end, spacing_mi):
    """Return monotonic bin edges between x_start and x_end with |spacing_mi| steps."""
    if spacing_mi == 0:
        raise ValueError("spacing_mi must be non-zero.")
    if x_start == x_end:
        raise ValueError("x_start and x_end must differ.")

    distance = x_end - x_start
    direction = np.sign(distance)
    step = abs(spacing_mi)
    num_segments = max(int(np.ceil(abs(distance) / step)), 1)

    edges = x_start + direction * step * np.arange(num_segments + 1)
    edges[-1] = x_end  # ensure exact end value
    return edges

def get_ramps_per_segment(ramps_path, x_start, x_end, spacing_mi):
    """Return boolean on/off-ramp flags per spatial bin."""
    ramps_df = pd.read_csv(ramps_path).copy()
    ramps_df["x"] = ramps_df["x_rcs_miles"]

    edges = _build_space_bins(x_start, x_end, spacing_mi)
    num_bins = len(edges) - 1
    on_ramp = np.zeros(num_bins, dtype=bool)
    off_ramp = np.zeros(num_bins, dtype=bool)

    x_vals = ramps_df["x"].to_numpy()
    entry_vals = ramps_df["entry_node"].astype(str).str.upper().eq("TRUE").to_numpy()
    exit_vals = ramps_df["exit_node"].astype(str).str.upper().eq("TRUE").to_numpy()

    asc_edges = edges if edges[0] < edges[-1] else edges[::-1]
    bin_idx = np.searchsorted(asc_edges, x_vals, side="right") - 1
    valid = (bin_idx >= 0) & (bin_idx < num_bins) & (x_vals >= asc_edges[0]) & (x_vals < asc_edges[-1])

    if edges[0] > edges[-1]:
        bin_idx = num_bins - 1 - bin_idx

    valid_bins = bin_idx[valid]
    on_ramp_counts = np.bincount(
        valid_bins,
        weights=entry_vals[valid].astype(np.uint8),
        minlength=num_bins,
    ).astype(int)
    off_ramp_counts = np.bincount(
        valid_bins,
        weights=exit_vals[valid].astype(np.uint8),
        minlength=num_bins,
    ).astype(int)
    on_ramp = on_ramp_counts > 0
    off_ramp = off_ramp_counts > 0

    for i, (on_count, off_count) in enumerate(zip(on_ramp_counts, off_ramp_counts)):
        low, high = sorted((edges[i], edges[i + 1]))
        print(
            f"Ramp bin {i} [{low}, {high}): "
            f"{on_count} on-ramps, {off_count} off-ramps"
        )

    return on_ramp, off_ramp, edges

def get_lanes_per_segment(lanes_path, x_start, x_end, spacing_mi, debug=False):
    """Return weighted-average lane counts per spatial bin."""
    lanes_df = pd.read_csv(lanes_path).copy()
    lanes_df["x_min"] = lanes_df[["x_start_mile", "x_end_mile"]].min(axis=1)
    lanes_df["x_max"] = lanes_df[["x_start_mile", "x_end_mile"]].max(axis=1)

    edges = _build_space_bins(x_start, x_end, spacing_mi)
    lane_vals = np.zeros(len(edges) - 1)
    if debug:
        direction = "ascending" if x_end > x_start else "descending"
        print(
            f"[lanes] x_start={x_start:.3f}, x_end={x_end:.3f}, spacing={abs(spacing_mi):.3f} mi, "
            f"direction={direction}, bins={len(edges)-1}"
        )
        print(f"[lanes] first 3 edges: {np.round(edges[:3], 4)}")
        print(f"[lanes] last  3 edges: {np.round(edges[-3:], 4)}")

    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        low, high = (a, b) if a < b else (b, a)
        overlapping = lanes_df[(lanes_df["x_max"] > low) & (lanes_df["x_min"] < high)]

        if overlapping.empty:
            lane_vals[i] = lane_vals[i - 1] if i > 0 else np.nan
            if debug:
                print(
                    f"[lanes][bin {i:02d}] range=({low:.3f}, {high:.3f}) overlap=0 -> "
                    f"lane={lane_vals[i]:.3f}"
                )
            continue

        overlap_len = np.minimum(overlapping["x_max"], high) - np.maximum(overlapping["x_min"], low)
        overlap_len = np.clip(overlap_len, 0, None)

        if np.any(overlap_len > 0):
            lane_vals[i] = np.average(overlapping["lanes"], weights=overlap_len)
            if debug:
                print(
                    f"[lanes][bin {i:02d}] range=({low:.3f}, {high:.3f}) overlap={len(overlapping)} "
                    f"weighted_lane={lane_vals[i]:.3f}"
                )
        else:
            lane_vals[i] = overlapping["lanes"].mean()
            if debug:
                print(
                    f"[lanes][bin {i:02d}] range=({low:.3f}, {high:.3f}) overlap={len(overlapping)} "
                    f"mean_lane={lane_vals[i]:.3f} (zero overlap lengths fallback)"
                )

    if debug:
        nan_count = int(np.isnan(lane_vals).sum())
        print(
            f"[lanes] done: min={np.nanmin(lane_vals):.3f}, max={np.nanmax(lane_vals):.3f}, "
            f"nan_count={nan_count}, first5={np.round(lane_vals[:5], 3)}"
        )
    return lane_vals

def fft_four_convs(Dp, Mp, k_cong, k_free, eps=1e-6, use_ortho=True):
    """
    FFT-accelerated 2D convolution of data and mask with two kernels.

    Parameters
    ----------
    Dp : Tensor (B, C, H, W) — data (NaN replaced with 0)
    Mp : Tensor (B, C, H, W) — binary mask (1 = observed, 0 = missing)
    k_cong, k_free : Tensor (F, C, Kh, Kw) — congestion / free-flow kernels
    eps : float — small constant added to denominator for numerical stability

    Returns
    -------
    sum_cong, N_cong, sum_free, N_free : Tensors (B, F, oh, ow)
        Weighted speed sums and weight counts for each kernel.
    """
    Dp = torch.nan_to_num(Dp, nan=0.0, posinf=0.0, neginf=0.0)
    Mp = torch.nan_to_num(Mp, nan=0.0, posinf=0.0, neginf=0.0)

    B, C, H, W = Dp.shape
    F_out, _, Kh, Kw = k_cong.shape
    Fh, Fw = H + Kh - 1, W + Kw - 1
    device, dtype = Dp.device, Dp.dtype

    Dp_pad = torch.zeros(B, C, Fh, Fw, device=device, dtype=dtype)
    Mp_pad = torch.zeros(B, C, Fh, Fw, device=device, dtype=dtype)
    Dp_pad[..., :H, :W] = Dp
    Mp_pad[..., :H, :W] = Mp

    k1_pad = torch.zeros(F_out, C, Fh, Fw, device=device, dtype=dtype)
    k2_pad = torch.zeros(F_out, C, Fh, Fw, device=device, dtype=dtype)
    k1_pad[..., :Kh, :Kw] = k_cong
    k2_pad[..., :Kh, :Kw] = k_free

    norm = "ortho" if use_ortho else None

    Df  = torch.fft.rfftn(Dp_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Mf  = torch.fft.rfftn(Mp_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Kf1 = torch.fft.rfftn(k1_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Kf2 = torch.fft.rfftn(k2_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)

    y1 = torch.fft.irfftn(Df * Kf1, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    y2 = torch.fft.irfftn(Df * Kf2, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    z1 = torch.fft.irfftn(Mf * Kf1, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    z2 = torch.fft.irfftn(Mf * Kf2, dim=(-2, -1), s=(Fh, Fw), norm=norm)

    oh, ow = H - Kh + 1, W - Kw + 1
    sum_cong = y1[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    sum_free = y2[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    N_cong   = z1[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow] + eps
    N_free   = z2[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow] + eps

    return sum_cong, N_cong, sum_free, N_free


class AdaptiveSmoothing(nn.Module):
    """
    ASMx: Adaptive Smoothing Method for Traffic State Estimation.

    Reconstructs a dense speed field from sparse detector observations by convolving
    with two physics-informed anisotropic kernels (congested and free-flow), blended
    via a smooth sigmoid transition.

    Parameters
    ----------
    kernel_time_window : float — half-extent of the kernel in time (seconds)
    kernel_space_window : float — half-extent of the kernel in space (miles)
    dx : float — spatial grid cell size (miles)
    dt : float — temporal grid cell size (seconds)
    init_delta : float — initial spatial smoothing scale (miles)
    init_tau : float — initial temporal smoothing scale (seconds)
    init_c_cong : float — congestion wave speed (mph)
    init_c_free : float — free-flow wave speed (mph)
    init_v_thr : float — speed threshold for blending (mph)
    init_v_delta : float — transition width for blending (mph)
    """
    def __init__(self,
                 kernel_time_window: float,
                 kernel_space_window: float,
                 dx: float,
                 dt: float,
                 init_delta: float = 0.09,
                 init_tau: float = 9.27,
                 init_c_cong: float = 12.26,
                 init_c_free: float = -50.40,
                 init_v_thr: float = 49.57,
                 init_v_delta: float = 10.11):
        super().__init__()
        self.size_t = int(kernel_time_window / dt)
        self.size_x = int(kernel_space_window / dx)
        self.dt = dt
        self.dx = dx

        t_offs = torch.arange(-self.size_t, self.size_t + 1) * dt
        x_offs = torch.arange(-self.size_x, self.size_x + 1) * dx
        X, T = torch.meshgrid(x_offs, t_offs, indexing='ij')
        self.register_buffer('T_offsets', T.float())
        self.register_buffer('X_offsets', X.float())

        self.delta   = nn.Parameter(torch.tensor(init_delta))
        self.tau     = nn.Parameter(torch.tensor(init_tau))
        self.c_cong  = nn.Parameter(torch.tensor(init_c_cong))
        self.c_free  = nn.Parameter(torch.tensor(init_c_free))
        self.v_thr   = nn.Parameter(torch.tensor(init_v_thr))
        self.v_delta = nn.Parameter(torch.tensor(init_v_delta))

    def forward(self, raw_data: torch.Tensor):
        """
        Run ASMx on a speed field.

        Parameters
        ----------
        raw_data : Tensor — 2D (space, time), 3D (batch, space, time),
                   or 4D (batch, channel, space, time). NaN = missing.

        Returns
        -------
        Tensor — reconstructed speed field (batch, space, time), no NaN.
        """
        if raw_data.ndim == 2:
            raw_data = raw_data.unsqueeze(0).unsqueeze(0)
        elif raw_data.ndim == 3:
            raw_data = raw_data.unsqueeze(1)

        mask = (~raw_data.isnan()).float()
        data = torch.nan_to_num(raw_data, nan=0.0)

        c_cong_s = self.c_cong / 3600.0
        c_free_s = self.c_free / 3600.0

        t_cong = self.T_offsets - self.X_offsets / c_cong_s
        t_free = self.T_offsets - self.X_offsets / c_free_s

        k_cong = torch.exp(-(t_cong.abs() / self.tau + self.X_offsets.abs() / self.delta))
        k_free = torch.exp(-(t_free.abs() / self.tau + self.X_offsets.abs() / self.delta))

        k_cong = k_cong.unsqueeze(0).unsqueeze(0)
        k_free = k_free.unsqueeze(0).unsqueeze(0)

        pad = (self.size_t, self.size_t, self.size_x, self.size_x)
        Dp = F.pad(data, pad, value=0.0)
        Mp = F.pad(mask, pad, value=0.0)

        sum_cong, N_cong, sum_free, N_free = fft_four_convs(Dp, Mp, k_cong, k_free)

        v_cong = sum_cong / N_cong
        v_free = sum_free / N_free

        v_min = torch.min(v_cong, v_free)
        w = 0.5 * (1 + torch.tanh((self.v_thr - v_min) / self.v_delta))
        v = w * v_cong + (1 - w) * v_free

        valid_cong = (N_cong > 0).float()
        valid_free = (N_free > 0).float()
        v = valid_cong * valid_free * v + (1 - valid_cong) * v_free + (1 - valid_free) * v_cong

        return v.squeeze(1)


def run_asmx(speed_matrix, dx, dt, delta, tau,
             c_cong=-13.0, c_free=45.0, v_thr=40.0, v_delta=10.0):
    """
    Convenience wrapper: run ASMx on a numpy speed matrix.

    Parameters
    ----------
    speed_matrix : ndarray (space, time) — speed in mph, NaN = missing
    dx, dt : float — spatial (mi) and temporal (s) resolution
    delta, tau : float — spatial and temporal smoothing scales
    c_cong, c_free, v_thr, v_delta : float — physics parameters

    Returns
    -------
    ndarray (space, time) — reconstructed speed field
    """
    space_size, time_size = speed_matrix.shape
    kernel_time_window = time_size * dt
    kernel_space_window = space_size * dx

    model = AdaptiveSmoothing(
        kernel_time_window=kernel_time_window,
        kernel_space_window=kernel_space_window,
        dx=dx, dt=dt,
        init_delta=delta, init_tau=tau,
        init_c_cong=c_cong, init_c_free=c_free,
        init_v_thr=v_thr, init_v_delta=v_delta
    )
    model.eval()

    speed_matrix = np.ascontiguousarray(speed_matrix)
    raw_tensor = torch.from_numpy(speed_matrix).float()
    with torch.no_grad():
        smoothed = model(raw_tensor)
    return smoothed[0].numpy()

# def average_neighbors_y(matrix, num_neighbors=3):
#     smoothed = matrix.astype(float, copy=True)
#     weight_template = np.arange(num_neighbors + 1, 0, -1)
#     full_weights = np.concatenate([weight_template[1:], weight_template])
#     for i in range(num_neighbors, matrix.shape[0] - num_neighbors):
#         for j in range(matrix.shape[1]):
#             window = matrix[i - num_neighbors : i + num_neighbors + 1, j]
#             valid_mask = ~np.isnan(window)
#             if not valid_mask.any():
#                 smoothed[i, j] = np.nan
#                 continue

#             smoothed[i, j] = np.average(
#                 window[valid_mask],
#                 weights=full_weights[valid_mask],
#             )
#     return smoothed


def y_weighted_fill_or_smooth(
    matrix,
    mode="impute",                 # "impute" or "smooth"
    num_neighbors=3,
    max_passes=5,
    include_center=True
):
    """
    mode="impute": only fill NaN cells using weighted y-neighbors.
    mode="smooth": update all cells using weighted y-neighbors.
    """
    if mode not in {"impute", "smooth"}:
        raise ValueError("mode must be either 'impute' or 'smooth'")

    # Force numeric ndarray with real np.nan
    arr = np.asarray(matrix)
    if not np.issubdtype(arr.dtype, np.number):
        arr = pd.DataFrame(arr).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        arr = arr.astype(float, copy=True)

    out = arr.copy()
    n_rows, n_cols = out.shape

    for _ in range(max_passes):
        prev_nan = np.isnan(out).sum()
        src = out.copy()  # read from previous pass only

        for i in range(n_rows):
            for j in range(n_cols):
                # In impute mode, skip non-NaN cells
                if mode == "impute" and not np.isnan(src[i, j]):
                    continue

                vals, wts = [], []

                # Only relevant for smooth mode
                if mode == "smooth" and include_center and not np.isnan(src[i, j]):
                    vals.append(src[i, j])
                    wts.append(num_neighbors + 1)

                for d in range(1, num_neighbors + 1):
                    w = num_neighbors - d + 1  # e.g., 3,2,1
                    up = i - d
                    dn = i + d

                    if up >= 0 and not np.isnan(src[up, j]):
                        vals.append(src[up, j]); wts.append(w)
                    if dn < n_rows and not np.isnan(src[dn, j]):
                        vals.append(src[dn, j]); wts.append(w)

                if vals:
                    out[i, j] = np.average(vals, weights=wts)
                elif mode == "smooth":
                    out[i, j] = np.nan  # smooth mode overwrites all cells

        # early stop for impute mode when no NaN count change
        if mode == "impute":
            new_nan = np.isnan(out).sum()
            if new_nan == prev_nan:
                break

    return out

def find_congestion_blocks_merged(matrix, threshold=35.0, connectivity=8, merge_touching=True):
    m = np.asarray(matrix, dtype=float)
    congested = np.isfinite(m) & (m > threshold)

    n_rows, n_cols = congested.shape
    visited = np.zeros_like(congested, dtype=bool)

    # neighbor definition
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

    # Step 1: connected components -> raw bounding boxes
    raw_boxes = []
    for r in range(n_rows):
        for c in range(n_cols):
            if not congested[r, c] or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            cells = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols:
                        if congested[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))

            rows = [x[0] for x in cells]
            cols = [x[1] for x in cells]
            raw_boxes.append({
                "row_min": min(rows),
                "row_max": max(rows),
                "col_min": min(cols),
                "col_max": max(cols),
                "n_cells": len(cells),
            })

    # Step 2: merge overlapping/touching bounding boxes
    def boxes_overlap(a, b, touching=False):
        if touching:
            row_disjoint = a["row_max"] < b["row_min"] - 1 or b["row_max"] < a["row_min"] - 1
            col_disjoint = a["col_max"] < b["col_min"] - 1 or b["col_max"] < a["col_min"] - 1
        else:
            row_disjoint = a["row_max"] < b["row_min"] or b["row_max"] < a["row_min"]
            col_disjoint = a["col_max"] < b["col_min"] or b["col_max"] < a["col_min"]
        return not (row_disjoint or col_disjoint)

    merged = raw_boxes[:]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            cur = merged[i].copy()
            used[i] = True

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if boxes_overlap(cur, merged[j], touching=merge_touching):
                    cur["row_min"] = min(cur["row_min"], merged[j]["row_min"])
                    cur["row_max"] = max(cur["row_max"], merged[j]["row_max"])
                    cur["col_min"] = min(cur["col_min"], merged[j]["col_min"])
                    cur["col_max"] = max(cur["col_max"], merged[j]["col_max"])
                    cur["n_cells"] += merged[j]["n_cells"]
                    used[j] = True
                    changed = True

            out.append(cur)
        merged = out

    # add corners
    for b in merged:
        b["corners"] = {
            "top_left": (b["row_min"], b["col_min"]),
            "top_right": (b["row_min"], b["col_max"]),
            "bottom_left": (b["row_max"], b["col_min"]),
            "bottom_right": (b["row_max"], b["col_max"]),
        }

    return merged

def filter_congestion_blocks(blocks, min_width_cols=6, min_height_rows=2):
    """
    Keep only blocks with:
      width  >= min_width_cols
      height >= min_height_rows
    """
    filtered = []
    for b in blocks:
        width = b["col_max"] - b["col_min"] + 1
        height = b["row_max"] - b["row_min"] + 1

        if width >= min_width_cols and height >= min_height_rows:
            b_out = dict(b)
            b_out["width_cols"] = width
            b_out["height_rows"] = height
            filtered.append(b_out)

    return filtered
