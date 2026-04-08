"""
Basic corridor schematic rendering from extracted lane and ramp CSV files.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Polygon, Rectangle
import pandas as pd


SPACE_BIN_MILES = 400.0 / 1609.344


def _coerce_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )


def load_lane_data(path: str | Path) -> pd.DataFrame:
    lanes = pd.read_csv(path).copy()
    lanes = lanes[["lanes", "x_start_mile", "x_end_mile"]].dropna(subset=["lanes", "x_start_mile", "x_end_mile"])
    lanes["lanes"] = pd.to_numeric(lanes["lanes"], errors="coerce")
    lanes["x_start_mile"] = pd.to_numeric(lanes["x_start_mile"], errors="coerce")
    lanes["x_end_mile"] = pd.to_numeric(lanes["x_end_mile"], errors="coerce")
    lanes = lanes.dropna().sort_values(["x_start_mile", "x_end_mile"]).reset_index(drop=True)
    lanes["lanes"] = lanes["lanes"].round().astype(int)
    # Some corridors are exported in descending postmile order. Normalize each
    # segment to [min_pm, max_pm] and let display order be handled separately.
    start_vals = lanes["x_start_mile"].copy()
    end_vals = lanes["x_end_mile"].copy()
    lanes["x_start_mile"] = start_vals.where(start_vals <= end_vals, end_vals)
    lanes["x_end_mile"] = end_vals.where(end_vals >= start_vals, start_vals)
    lanes = lanes.sort_values(["x_start_mile", "x_end_mile"]).reset_index(drop=True)
    return lanes


def _merge_adjacent_same_lanes(lanes: pd.DataFrame) -> pd.DataFrame:
    if lanes.empty:
        return lanes.copy()

    records: list[dict] = []
    current = lanes.iloc[0].to_dict()
    for _, row in lanes.iloc[1:].iterrows():
        if int(row["lanes"]) == int(current["lanes"]) and abs(float(row["x_start_mile"]) - float(current["x_end_mile"])) <= 1e-6:
            current["x_end_mile"] = float(row["x_end_mile"])
        else:
            records.append(current.copy())
            current = row.to_dict()
    records.append(current.copy())
    return pd.DataFrame(records)


def simplify_lane_segments(lanes: pd.DataFrame, min_persistent_miles: float = 0.25) -> pd.DataFrame:
    lanes = lanes.sort_values(["x_start_mile", "x_end_mile"]).reset_index(drop=True).copy()
    if len(lanes) < 2:
        return _merge_adjacent_same_lanes(lanes)

    changed = True
    while changed and len(lanes) >= 2:
        changed = False
        for idx in range(len(lanes)):
            seg = lanes.iloc[idx]
            seg_len = float(seg["x_end_mile"]) - float(seg["x_start_mile"])
            if seg_len >= min_persistent_miles:
                continue

            if idx == 0:
                lanes.loc[1, "x_start_mile"] = float(seg["x_start_mile"])
                lanes = lanes.drop(index=[0]).reset_index(drop=True)
            elif idx == len(lanes) - 1:
                lanes.loc[idx - 1, "x_end_mile"] = float(seg["x_end_mile"])
                lanes = lanes.drop(index=[idx]).reset_index(drop=True)
            else:
                prev_seg = lanes.iloc[idx - 1]
                next_seg = lanes.iloc[idx + 1]
                prev_len = float(prev_seg["x_end_mile"]) - float(prev_seg["x_start_mile"])
                next_len = float(next_seg["x_end_mile"]) - float(next_seg["x_start_mile"])

                if int(prev_seg["lanes"]) == int(next_seg["lanes"]):
                    lanes.loc[idx - 1, "x_end_mile"] = float(next_seg["x_end_mile"])
                    lanes = lanes.drop(index=[idx, idx + 1]).reset_index(drop=True)
                elif prev_len >= next_len:
                    lanes.loc[idx - 1, "x_end_mile"] = float(seg["x_end_mile"])
                    lanes = lanes.drop(index=[idx]).reset_index(drop=True)
                else:
                    lanes.loc[idx + 1, "x_start_mile"] = float(seg["x_start_mile"])
                    lanes = lanes.drop(index=[idx]).reset_index(drop=True)
            changed = True
            break

    return _merge_adjacent_same_lanes(lanes)


def load_ramp_data(path: str | Path) -> pd.DataFrame:
    ramps = pd.read_csv(path).copy()
    ramps["x_rcs_miles"] = pd.to_numeric(ramps.get("x_rcs_miles"), errors="coerce")
    ramps["num_lanes"] = pd.to_numeric(ramps.get("num_lanes"), errors="coerce")
    ramps["entry_node"] = _coerce_bool(ramps.get("entry_node", pd.Series(dtype=object)))
    ramps["exit_node"] = _coerce_bool(ramps.get("exit_node", pd.Series(dtype=object)))
    ramps = ramps.dropna(subset=["x_rcs_miles"]).sort_values("x_rcs_miles").reset_index(drop=True)
    return ramps


def _parse_corridor_name(name: str) -> tuple[int, str]:
    match = re.match(r"^[A-Z]+(\d+)_([NSEW])$", name.upper())
    if not match:
        raise ValueError(f"Unable to infer freeway/direction from corridor name '{name}'.")
    return int(match.group(1)), match.group(2)


def load_detector_data(
    path: str | Path,
    *,
    corridor_name: str,
    x_min: float | None = None,
    x_max: float | None = None,
) -> pd.DataFrame:
    fwy, direction = _parse_corridor_name(corridor_name)
    detectors = pd.read_csv(path, sep="\t").copy()
    detectors["Fwy"] = pd.to_numeric(detectors["Fwy"], errors="coerce")
    detectors["Abs_PM"] = pd.to_numeric(detectors["Abs_PM"], errors="coerce")
    detectors["Dir"] = detectors["Dir"].astype(str).str.upper().str.strip()
    detectors["Type"] = detectors["Type"].astype(str).str.upper().str.strip()
    detectors = detectors[
        (detectors["Fwy"] == fwy)
        & (detectors["Dir"] == direction)
        & (detectors["Type"] == "ML")
    ].dropna(subset=["Abs_PM"])
    if x_min is not None:
        detectors = detectors[detectors["Abs_PM"] >= x_min]
    if x_max is not None:
        detectors = detectors[detectors["Abs_PM"] <= x_max]
    detectors = detectors.sort_values("Abs_PM").reset_index(drop=True)
    return detectors


def _layout_ramps(ramps: pd.DataFrame, default_dx: float) -> pd.DataFrame:
    if ramps.empty:
        ramps = ramps.copy()
        ramps["layout_level"] = pd.Series(dtype=int)
        ramps["layout_dx"] = pd.Series(dtype=float)
        return ramps

    ramps = ramps.sort_values("x_rcs_miles").reset_index(drop=True).copy()
    stations = ramps["x_rcs_miles"].astype(float).tolist()
    dx_values: list[float] = []
    for idx, x in enumerate(stations):
        prev_gap = x - stations[idx - 1] if idx > 0 else None
        next_gap = stations[idx + 1] - x if idx < len(stations) - 1 else None
        neighbor_gaps = [gap for gap in (prev_gap, next_gap) if gap is not None and gap > 0]
        local_dx = default_dx
        if neighbor_gaps:
            local_dx = min(local_dx, max(0.08, min(neighbor_gaps) * 0.42))
        dx_values.append(local_dx)

    levels_end_x: list[float] = []
    assigned_levels: list[int] = []
    for x, dx in zip(stations, dx_values):
        start_x = x - dx
        level = 0
        while level < len(levels_end_x) and start_x <= levels_end_x[level]:
            level += 1
        if level == len(levels_end_x):
            levels_end_x.append(x + dx)
        else:
            levels_end_x[level] = x + dx
        assigned_levels.append(level)

    ramps["layout_level"] = assigned_levels
    ramps["layout_dx"] = dx_values
    return ramps


def save_legend_figure(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(4.8, 1.2))
    ax.axis("off")
    road_fill = "#f4f1e8"
    boundary_color = "#2f3640"
    legend_handles = [
        Polygon([(0, 0), (1, 0), (1, 0.4), (0, 0.4)], closed=True, facecolor=road_fill, edgecolor=boundary_color, label="Mainline"),
        plt.Line2D([0], [0], color="#1f9d55", lw=3.6, label="Inflow"),
        plt.Line2D([0], [0], color="#d64545", lw=3.6, label="Outflow"),
        plt.Line2D([0], [0], color="#2563eb", lw=1.2, label="Detector"),
    ]
    ax.legend(handles=legend_handles, loc="center", ncol=4, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output_path


def configure_font(font_path: str | Path | None) -> str | None:
    if not font_path:
        return None
    font_path = str(font_path)
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    family_name = prop.get_name()
    plt.rcParams["font.family"] = family_name
    return family_name


def render_schematic(
    lanes: pd.DataFrame,
    ramps: pd.DataFrame,
    output_path: str | Path,
    *,
    detectors: pd.DataFrame | None = None,
    title: str | None = None,
    station_tick_miles: float = 1.0,
    indices_per_figure: int = 20,
    display_descending: bool = False,
    corridor_start_mile: float | None = None,
    corridor_end_mile: float | None = None,
) -> list[Path]:
    if lanes.empty:
        raise ValueError("Lane dataframe is empty.")

    lanes = simplify_lane_segments(lanes)
    output_path = Path(output_path)
    data_min = float(lanes["x_start_mile"].min())
    data_max = float(lanes["x_end_mile"].max())
    x_min = data_min if corridor_start_mile is None or corridor_end_mile is None else min(corridor_start_mile, corridor_end_mile)
    x_max = data_max if corridor_start_mile is None or corridor_end_mile is None else max(corridor_start_mile, corridor_end_mile)
    max_lanes = int(lanes["lanes"].max())
    corridor_length = max(x_max - x_min, 0.5)
    bin_span = SPACE_BIN_MILES
    bins_total = max(1, math.ceil(corridor_length / bin_span))
    indices_per_figure = max(1, indices_per_figure)
    num_windows = max(1, math.ceil(bins_total / indices_per_figure))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    base_stem = output_path.stem
    suffix = output_path.suffix or ".png"

    for window_idx in range(num_windows):
        if display_descending:
            first_bin_idx = window_idx * indices_per_figure
            last_bin_exclusive = min((window_idx + 1) * indices_per_figure, bins_total)
            window_end = x_max - first_bin_idx * bin_span
            window_start = max(x_max - last_bin_exclusive * bin_span, x_min)
        else:
            first_bin_idx = window_idx * indices_per_figure
            last_bin_exclusive = min((window_idx + 1) * indices_per_figure, bins_total)
            window_start = x_min + first_bin_idx * bin_span
            window_end = min(x_min + last_bin_exclusive * bin_span, x_max)
        lanes_window = lanes[
            (lanes["x_end_mile"] > window_start) & (lanes["x_start_mile"] < window_end)
        ].copy()
        ramps_window = ramps[
            (ramps["x_rcs_miles"] >= window_start) & (ramps["x_rcs_miles"] <= window_end)
        ].copy()
        if detectors is None:
            detectors_window = pd.DataFrame(columns=["Abs_PM"])
        else:
            detectors_window = detectors[
                (detectors["Abs_PM"] >= window_start) & (detectors["Abs_PM"] <= window_end)
            ].copy()

        fig_width = min(max(12.0, (window_end - window_start) * 1.1), 16.0)
        fig_height = min(max(2.8, 1.8 + max_lanes * 0.34), 6.2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        road_fill = "#f4f1e8"
        boundary_color = "#2f3640"
        separator_color = "#7b8490"
        lane_width = 1.0
        def lane_count_at(station: float) -> int:
            containing = lanes[
                (lanes["x_start_mile"] <= station + 1e-9) & (lanes["x_end_mile"] > station - 1e-9)
            ]
            if containing.empty:
                prior = lanes[lanes["x_end_mile"] <= station]
                if not prior.empty:
                    return int(prior.iloc[-1]["lanes"])
                return int(lanes.iloc[0]["lanes"])
            return int(containing.iloc[0]["lanes"])

        centerline_segments: list[tuple[float, float, float]] = []
        for row in lanes_window.itertuples(index=False):
            left = max(window_start, float(row.x_start_mile))
            right = min(window_end, float(row.x_end_mile))
            if right - left <= 1e-6:
                continue
            road_top = int(row.lanes) * lane_width
            centerline_segments.append((left, right, road_top))
            ax.add_patch(
                Rectangle(
                    (left, 0.0),
                    right - left,
                    road_top,
                    facecolor=road_fill,
                    edgecolor="none",
                    zorder=1,
                )
            )

        def _collect_intervals(predicate) -> list[tuple[float, float]]:
            intervals: list[tuple[float, float]] = []
            for left, right, road_top in centerline_segments:
                if predicate(road_top):
                    if intervals and abs(intervals[-1][1] - left) <= 1e-6:
                        intervals[-1] = (intervals[-1][0], right)
                    else:
                        intervals.append((left, right))
            return intervals

        ax.plot([window_start, window_end], [0.0, 0.0], color=boundary_color, linewidth=1.8, zorder=4)

        for lane_idx in range(1, max_lanes):
            y = lane_idx * lane_width
            intervals = _collect_intervals(lambda road_top, y=y: road_top >= y - 1e-6)
            if not intervals:
                continue
            for left, right in intervals:
                ax.plot(
                    [left, right],
                    [y, y],
                    color=separator_color,
                    linewidth=1.1,
                    linestyle=(0, (2.2, 2.2)),
                    zorder=4,
                )

        for left, right, road_top in centerline_segments:
            ax.plot([left, right], [road_top, road_top], color=boundary_color, linewidth=1.8, zorder=5)

        ramps_window = _layout_ramps(ramps_window, max(0.12, min(0.22, (window_end - window_start) * 0.02)))
        ramp_tick_length = 0.34
        max_road_top = max((road_top for _, _, road_top in centerline_segments), default=max_lanes * lane_width)
        bracket_gap = 0.1
        bracket_base_y = max_road_top + ramp_tick_length + bracket_gap
        bracket_top_y = bracket_base_y + 0.18

        def top_boundary_at(station: float) -> float:
            return lane_count_at(station) * lane_width

        for _, row in ramps_window.iterrows():
            x = float(row["x_rcs_miles"])
            road_top = top_boundary_at(x)
            tick_base_y = road_top
            tick_top_y = road_top + ramp_tick_length
            if row["entry_node"]:
                ax.plot([x, x], [tick_base_y, tick_top_y], color="#1f9d55", linewidth=3.6, zorder=7)
            if row["exit_node"]:
                ax.plot([x, x], [tick_base_y, tick_top_y], color="#d64545", linewidth=3.6, zorder=7)

        for _, row in detectors_window.iterrows():
            x = float(row["Abs_PM"])
            road_top = top_boundary_at(x)
            ax.plot([x, x], [0.0, road_top], color="#2563eb", linewidth=1.2, zorder=6)

        for bin_idx in range(first_bin_idx, last_bin_exclusive):
            if display_descending:
                bin_right = x_max - bin_idx * bin_span
                bin_left = max(x_max - (bin_idx + 1) * bin_span, x_min)
                display_index = bin_idx + 1
            else:
                bin_left = x_min + bin_idx * bin_span
                bin_right = min(x_min + (bin_idx + 1) * bin_span, x_max)
                display_index = bin_idx + 1
            if bin_right <= window_start or bin_left >= window_end:
                continue
            left = max(window_start, bin_left)
            right = min(window_end, bin_right)
            ax.plot([left, right], [bracket_base_y, bracket_base_y], color="#4b5563", linewidth=1.1, zorder=8)
            ax.plot([left, left], [bracket_base_y, bracket_top_y], color="#4b5563", linewidth=1.1, zorder=8)
            ax.plot([right, right], [bracket_base_y, bracket_top_y], color="#4b5563", linewidth=1.1, zorder=8)
            ax.text(
                (left + right) / 2.0,
                bracket_top_y + 0.05,
                f"s={display_index}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#374151",
                zorder=8,
            )

        tick_start = int(window_start // station_tick_miles) * station_tick_miles
        tick_end = window_end + station_tick_miles
        xticks = []
        x = tick_start
        while x <= tick_end:
            if window_start <= x <= window_end:
                xticks.append(round(x, 6))
            x += station_tick_miles

        if display_descending:
            ax.set_xlim(window_end, window_start)
        else:
            ax.set_xlim(window_start, window_end)
        ax.set_ylim(-0.25, bracket_top_y + 0.38)
        ax.set_xticks(xticks)
        ax.set_xlabel("Postmile / Station")
        ax.set_yticks([])
        ax.grid(axis="x", linestyle="--", color="#e5e7eb", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        if title:
            if num_windows == 1:
                ax.set_title(title)
            else:
                ax.set_title(
                    f"{title} (s={first_bin_idx + 1}-{last_bin_exclusive}, {window_start:.2f}-{window_end:.2f} mi)"
                )

        fig.tight_layout()
        if num_windows == 1:
            window_output_path = output_path
        else:
            window_output_path = output_path.with_name(f"{base_stem}_part{window_idx + 1}{suffix}")
        fig.savefig(window_output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(window_output_path)

    return saved_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a basic corridor schematic from lane/ramp CSVs.")
    parser.add_argument("--lanes", required=True, help="Path to lanes.csv or lanes_validated.csv")
    parser.add_argument("--ramps", required=True, help="Path to ramps.csv or ramps_validated.csv")
    parser.add_argument("--output", required=True, help="Output image path, e.g. schematic.svg or schematic.png")
    parser.add_argument("--legend-output", default=None, help="Optional standalone legend image path")
    parser.add_argument("--detectors", default=None, help="Optional station metadata file to render ML detectors")
    parser.add_argument("--corridor-name", default=None, help="Corridor id like I405_N; inferred from lanes path parent if omitted")
    parser.add_argument("--font-path", default=None, help="Optional font file to use for all figure text")
    parser.add_argument("--title", default=None, help="Optional title")
    parser.add_argument("--station-tick-miles", type=float, default=1.0, help="Station tick spacing in miles")
    parser.add_argument("--indices-per-figure", type=int, default=20, help="Number of 400 m cells per figure")
    parser.add_argument("--range-start", type=float, default=None, help="Optional start postmile for subsetting")
    parser.add_argument("--range-end", type=float, default=None, help="Optional end postmile for subsetting")
    parser.add_argument("--display-descending", action="store_true", help="Display the x-axis in descending postmile order")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_font(args.font_path)
    lanes = load_lane_data(args.lanes)
    ramps = load_ramp_data(args.ramps)
    if args.range_start is not None and args.range_end is not None:
        subset_min = min(args.range_start, args.range_end)
        subset_max = max(args.range_start, args.range_end)
        lanes = lanes[(lanes["x_end_mile"] > subset_min) & (lanes["x_start_mile"] < subset_max)].copy()
        ramps = ramps[(ramps["x_rcs_miles"] >= subset_min) & (ramps["x_rcs_miles"] <= subset_max)].copy()
        if lanes.empty:
            raise ValueError("No lane records found in the requested postmile range.")
    corridor_name = args.corridor_name or Path(args.lanes).resolve().parent.name
    detectors = None
    if args.detectors:
        detector_min = float(lanes["x_start_mile"].min()) if args.range_start is None else min(args.range_start, args.range_end)
        detector_max = float(lanes["x_end_mile"].max()) if args.range_start is None else max(args.range_start, args.range_end)
        detectors = load_detector_data(
            args.detectors,
            corridor_name=corridor_name,
            x_min=detector_min,
            x_max=detector_max,
        )
    render_schematic(
        lanes,
        ramps,
        args.output,
        detectors=detectors,
        title=args.title,
        station_tick_miles=args.station_tick_miles,
        indices_per_figure=args.indices_per_figure,
        display_descending=args.display_descending,
        corridor_start_mile=args.range_start,
        corridor_end_mile=args.range_end,
    )
    if args.legend_output:
        save_legend_figure(args.legend_output)


if __name__ == "__main__":
    main()
