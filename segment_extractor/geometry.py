"""
Geometric helpers for clipping, stationing, and aggregation.
"""
from __future__ import annotations

from typing import List, Tuple, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, nearest_points

StationingDirection = Literal["ascending", "descending"]


def _station_from_anchor(
    distance_ft: float,
    anchor_distance_ft: float,
    anchor_postmile: float,
    stationing_direction: StationingDirection,
) -> float:
    """
    Convert a projected distance along the reference line into a station value.
    """
    if stationing_direction not in {"ascending", "descending"}:
        raise ValueError(f"Invalid stationing direction: {stationing_direction}")

    anchor_ft = anchor_postmile * 5280.0
    delta_ft = distance_ft - anchor_distance_ft
    if stationing_direction == "ascending":
        return anchor_ft + delta_ft
    return anchor_ft - delta_ft


def _station_from_two_point_calibration(
    distance_ft: float,
    total_length_ft: float,
    start_postmile: float,
    end_postmile: float,
) -> float:
    """
    Map distance-along-line to station via linear interpolation between
    start and end postmile anchors.
    """
    if total_length_ft <= 0:
        raise ValueError("Total corridor length must be positive for two-point calibration.")

    start_ft = start_postmile * 5280.0
    end_ft = end_postmile * 5280.0
    ratio = max(0.0, min(1.0, distance_ft / total_length_ft))
    return start_ft + ratio * (end_ft - start_ft)


def to_state_plane(gdf_or_series, projection):
    """
    Project a GeoDataFrame/GeoSeries to the specified CRS.
    """
    if hasattr(gdf_or_series, "to_crs"):
        return gdf_or_series.to_crs(projection)
    raise TypeError("Object must be a GeoDataFrame/GeoSeries with a .to_crs method")


def clip_line_between_coords(gdf, start_coord, end_coord, projection):
    """
    Clip the ordered GeoDataFrame between arbitrary coordinates.
    """
    gdf_proj = gdf.to_crs(projection).reset_index(drop=True).copy()
    non_empty = gdf_proj.geometry.notnull() & (~gdf_proj.geometry.is_empty)
    if not non_empty.all():
        gdf_proj = gdf_proj[non_empty].reset_index(drop=True)
    if gdf_proj.empty:
        raise ValueError("No valid path geometry available for clipping.")
    start_pt = gpd.GeoSeries([Point(start_coord)], crs="EPSG:4326").to_crs(projection).iloc[0]
    end_pt = gpd.GeoSeries([Point(end_coord)], crs="EPSG:4326").to_crs(projection).iloc[0]

    dist_start = gdf_proj.geometry.distance(start_pt)
    dist_end = gdf_proj.geometry.distance(end_pt)
    start_idx = int(dist_start.idxmin())
    end_idx = int(dist_end.idxmin())

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
        start_pt, end_pt = end_pt, start_pt

    seg_start = gdf_proj.geometry.iloc[start_idx]
    d_snap_start = seg_start.project(start_pt)
    snap_start = seg_start.interpolate(d_snap_start)

    new_start_coords = []
    for i in range(len(seg_start.coords) - 1):
        p0 = Point(seg_start.coords[i])
        p1 = Point(seg_start.coords[i + 1])
        d0 = seg_start.project(p0)
        d1 = seg_start.project(p1)
        if d0 <= d_snap_start <= d1 or d1 <= d_snap_start <= d0:
            new_start_coords.append((snap_start.x, snap_start.y))
            new_start_coords.append((p1.x, p1.y))
        elif d0 > d_snap_start and d1 > d_snap_start:
            new_start_coords.append((p1.x, p1.y))
    if len(new_start_coords) < 2:
        trimmed_start = LineString([])
    else:
        cleaned = [new_start_coords[0]]
        for c in new_start_coords[1:]:
            if c != cleaned[-1]:
                cleaned.append(c)
        trimmed_start = LineString(cleaned)

    seg_end = gdf_proj.geometry.iloc[end_idx]
    d_snap_end = seg_end.project(end_pt)
    snap_end = seg_end.interpolate(d_snap_end)

    new_end_coords = []
    coords = list(seg_end.coords)
    for i in range(len(coords) - 1):
        p0 = Point(coords[i])
        p1 = Point(coords[i + 1])
        d0 = seg_end.project(p0)
        d1 = seg_end.project(p1)
        if (d0 <= d_snap_end <= d1) or (d1 <= d_snap_end <= d0):
            new_end_coords.append((p0.x, p0.y))
            new_end_coords.append((snap_end.x, snap_end.y))
            break
        elif d0 <= d_snap_end and d1 <= d_snap_end:
            new_end_coords.append((p0.x, p0.y))
    if not new_end_coords or new_end_coords[-1] != (snap_end.x, snap_end.y):
        new_end_coords.append((snap_end.x, snap_end.y))
    if len(new_end_coords) < 2:
        trimmed_end = LineString([])
    else:
        cleaned = [new_end_coords[0]]
        for c in new_end_coords[1:]:
            if c != cleaned[-1]:
                cleaned.append(c)
        trimmed_end = LineString(cleaned)

    clipped = gdf_proj.iloc[start_idx : end_idx + 1].copy()
    clipped.geometry.iloc[0] = trimmed_start
    clipped.geometry.iloc[-1] = trimmed_end
    clipped = clipped[clipped.geometry.length > 0]
    return clipped.to_crs("EPSG:4326")


def fit_reference_spline(
    line: LineString,
    projection: str,
    smoothing: float = 100.0,
    n_points: int = 2000,
    plot: bool = True,
):
    """
    Fit a spline through a WGS84 LineString.
    """
    if line.is_empty:
        raise ValueError("LineString is empty.")

    transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
    xs, ys = transformer.transform(*zip(*list(line.coords)))

    tck, u = splprep([xs, ys], s=smoothing)
    unew = np.linspace(0, 1, n_points)
    out = splev(unew, tck)

    transformer_back = Transformer.from_crs(projection, "EPSG:4326", always_xy=True)
    lon, lat = transformer_back.transform(out[0], out[1])
    smoothed = LineString(zip(lon, lat))

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(xs, ys, "k--", label="Original")
        ax.plot(out[0], out[1], "r", label="Spline")
        ax.legend()
        ax.set_aspect("equal")
        plt.show()

    return smoothed, tck


def merge_segments_to_line(gdf):
    """
    Merge multiple LineString rows into a single LineString.
    """
    merged = linemerge(gdf.geometry.unary_union)
    if merged.geom_type == "MultiLineString":
        merged = max(merged.geoms, key=lambda ls: ls.length)
    return merged


def snap_point_to_lines(gdf_lines: gpd.GeoDataFrame, lat: float, lon: float) -> Tuple[float, float]:
    """
    Snap an arbitrary lat/lon to the nearest position along the provided linework GeoDataFrame.
    """
    if gdf_lines.empty:
        raise ValueError("No roadway geometry available for snapping.")

    lines_3857 = gdf_lines.to_crs(epsg=3857)
    if lines_3857.empty:
        raise ValueError("No roadway geometry available for snapping.")

    point = (
        gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
        .to_crs(epsg=3857)
        .iloc[0]
    )
    union = lines_3857.unary_union
    if union.is_empty:
        raise ValueError("Unable to snap to roadway geometry.")

    snapped = nearest_points(point, union)[1]
    snapped_wgs = (
        gpd.GeoSeries([snapped], crs="EPSG:3857")
        .to_crs(epsg=4326)
        .iloc[0]
    )
    return float(snapped_wgs.y), float(snapped_wgs.x)


def compute_point_stationing(
    gdf_points: gpd.GeoDataFrame,
    gdf_trimmed: gpd.GeoDataFrame,
    anchor_coord: Tuple[float, float],
    anchor_postmile: float,
    projection: str,
    segment_length_mi: float,
    stationing_direction: StationingDirection = "ascending",
    end_postmile: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Compute stationing (feet & miles) for arbitrary points along a trimmed roadway.
    """
    gdf_trimmed_proj = gdf_trimmed.to_crs(projection)
    gdf_proj = gdf_points.to_crs(projection).copy()
    ref_line = merge_segments_to_line(gdf_trimmed_proj)

    anchor_pt = gpd.GeoSeries([Point(anchor_coord)], crs="EPSG:4326").to_crs(projection).iloc[0]
    d_anchor = ref_line.project(anchor_pt)
    total_length_ft = float(ref_line.length)

    distances_ft = []
    stations_ft = []
    for geom in gdf_proj.geometry:
        d_point = ref_line.project(geom)
        distances_ft.append(d_point)
        if end_postmile is not None:
            stations_ft.append(
                _station_from_two_point_calibration(
                    distance_ft=d_point,
                    total_length_ft=total_length_ft,
                    start_postmile=anchor_postmile,
                    end_postmile=end_postmile,
                )
            )
        else:
            stations_ft.append(
                _station_from_anchor(
                    distance_ft=d_point,
                    anchor_distance_ft=d_anchor,
                    anchor_postmile=anchor_postmile,
                    stationing_direction=stationing_direction,
                )
            )

    gdf_proj["distance_ft"] = distances_ft
    gdf_proj["distance_miles"] = gdf_proj["distance_ft"] / 5280.0
    gdf_proj["x_rcs"] = stations_ft
    gdf_proj["x_rcs_miles"] = gdf_proj["x_rcs"] / 5280.0
    gdf_proj = gdf_proj.sort_values("distance_ft")
    gdf_proj = gdf_proj[
        (gdf_proj["distance_miles"] != 0) & (gdf_proj["distance_miles"] != segment_length_mi)
    ]
    return gdf_proj.to_crs("EPSG:4326")


def compute_segment_stationing(
    gdf_segments: gpd.GeoDataFrame,
    gdf_trimmed: gpd.GeoDataFrame,
    anchor_coord: Tuple[float, float],
    anchor_postmile: float,
    projection: str,
    stationing_direction: StationingDirection = "ascending",
    end_postmile: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Compute stationing for each segment in the trimmed corridor.
    """
    gdf_trimmed_proj = gdf_trimmed.to_crs(projection)
    gdf_proj = gdf_segments.to_crs(projection).copy()
    ref_line = merge_segments_to_line(gdf_trimmed_proj)

    anchor_pt = gpd.GeoSeries([Point(anchor_coord)], crs="EPSG:4326").to_crs(projection).iloc[0]
    d_anchor = ref_line.project(anchor_pt)
    total_length_ft = float(ref_line.length)

    x_start_ft = []
    x_end_ft = []
    distance_start_ft = []
    distance_end_ft = []
    for geom in gdf_proj.geometry:
        start_pt, end_pt = Point(geom.coords[0]), Point(geom.coords[-1])
        d_start = ref_line.project(start_pt)
        d_end = ref_line.project(end_pt)
        distance_start_ft.append(d_start)
        distance_end_ft.append(d_end)
        if end_postmile is not None:
            x_start_ft.append(
                _station_from_two_point_calibration(
                    distance_ft=d_start,
                    total_length_ft=total_length_ft,
                    start_postmile=anchor_postmile,
                    end_postmile=end_postmile,
                )
            )
            x_end_ft.append(
                _station_from_two_point_calibration(
                    distance_ft=d_end,
                    total_length_ft=total_length_ft,
                    start_postmile=anchor_postmile,
                    end_postmile=end_postmile,
                )
            )
        else:
            x_start_ft.append(
                _station_from_anchor(
                    distance_ft=d_start,
                    anchor_distance_ft=d_anchor,
                    anchor_postmile=anchor_postmile,
                    stationing_direction=stationing_direction,
                )
            )
            x_end_ft.append(
                _station_from_anchor(
                    distance_ft=d_end,
                    anchor_distance_ft=d_anchor,
                    anchor_postmile=anchor_postmile,
                    stationing_direction=stationing_direction,
                )
            )

    gdf_proj["x_start_ft"] = x_start_ft
    gdf_proj["x_end_ft"] = x_end_ft
    gdf_proj["x_start_mile"] = gdf_proj["x_start_ft"] / 5280.0
    gdf_proj["x_end_mile"] = gdf_proj["x_end_ft"] / 5280.0
    gdf_proj["distance_start_ft"] = distance_start_ft
    gdf_proj["distance_end_ft"] = distance_end_ft
    gdf_proj["distance_start_mile"] = gdf_proj["distance_start_ft"] / 5280.0
    gdf_proj["distance_end_mile"] = gdf_proj["distance_end_ft"] / 5280.0
    gdf_out = gdf_proj.to_crs("EPSG:4326")

    gdf_out["start_node_lon"] = gdf_out.geometry.apply(lambda g: g.coords[0][0])
    gdf_out["start_node_lat"] = gdf_out.geometry.apply(lambda g: g.coords[0][1])
    gdf_out.geometry = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(gdf_out["start_node_lon"], gdf_out["start_node_lat"])],
        crs="EPSG:4326",
    )
    return gdf_out


def aggregate_lane_blocks(gdf_segments: gpd.GeoDataFrame, max_gap_ft: float = 500.0) -> gpd.GeoDataFrame:
    """
    Merge segments that share the same lane count and are contiguous.
    """
    has_distance = {"distance_start_ft", "distance_end_ft"} <= set(gdf_segments.columns)
    sort_col = "distance_start_ft" if has_distance else "x_start_ft"
    distance_start_col = "distance_start_ft" if has_distance else "x_start_ft"
    distance_end_col = "distance_end_ft" if has_distance else "x_end_ft"

    df = gdf_segments.sort_values(sort_col).reset_index(drop=True)
    out = []
    current = None

    for _, row in df.iterrows():
        if current is None:
            current = {
                "lanes": row["lanes"],
                "x_start_ft": row["x_start_ft"],
                "x_end_ft": row["x_end_ft"],
                "distance_start_ft": row[distance_start_col],
                "distance_end_ft": row[distance_end_col],
                "start_node_lat": row["start_node_lat"],
                "start_node_lon": row["start_node_lon"],
            }
            continue

        same_lanes = row["lanes"] == current["lanes"]
        touching_or_close = abs(current["distance_end_ft"] - row[distance_start_col]) <= max_gap_ft

        if same_lanes and touching_or_close:
            current["x_end_ft"] = row["x_end_ft"]
            current["distance_end_ft"] = row[distance_end_col]
        else:
            out.append(current)
            current = {
                "lanes": row["lanes"],
                "x_start_ft": row["x_start_ft"],
                "x_end_ft": row["x_end_ft"],
                "distance_start_ft": row[distance_start_col],
                "distance_end_ft": row[distance_end_col],
                "start_node_lat": row["start_node_lat"],
                "start_node_lon": row["start_node_lon"],
            }

    if current:
        out.append(current)

    df_out = pd.DataFrame(out)
    df_out["x_start_mile"] = df_out["x_start_ft"] / 5280.0
    df_out["x_end_mile"] = df_out["x_end_ft"] / 5280.0
    df_out["length_mi"] = (
        (df_out["distance_end_ft"] - df_out["distance_start_ft"]).abs() / 5280.0
        if has_distance
        else (df_out["x_end_ft"] - df_out["x_start_ft"]) / 5280.0
    )
    return gpd.GeoDataFrame(
        df_out,
        geometry=gpd.points_from_xy(df_out["start_node_lon"], df_out["start_node_lat"]),
        crs="EPSG:4326",
    )[
        ["lanes", "x_start_mile", "x_end_mile", "length_mi", "start_node_lat", "start_node_lon", "geometry"]
    ]
