"""
GMNS export helpers for extracted corridor networks.

Assumptions for this interoperability pass:
- The corridor mainline remains the backbone of the exported network.
- Mainline links are split at ramp attachment points so ramp-related nodes are
  explicit in `node.csv`.
- Node IDs and link IDs are synthesized as integers because clipped endpoints
  and inserted ramp nodes may not align to existing OSM IDs.
- Geometry is stored in WKT with coordinates in EPSG:4326.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import substring, transform as shapely_transform

GMNS_VERSION = 0.97
WGS84 = "EPSG:4326"
NODE_COLUMNS = [
    "node_id",
    "name",
    "x_coord",
    "y_coord",
    "z_coord",
    "node_type",
    "ctrl_type",
    "zone_id",
    "parent_node_id",
]
LINK_COLUMNS = [
    "link_id",
    "name",
    "from_node_id",
    "to_node_id",
    "directed",
    "geometry",
    "dir_flag",
    "length",
    "facility_type",
    "free_speed",
    "lanes",
    "osm_way_id",
    "ref",
]
CONFIG_COLUMNS = [
    "dataset_name",
    "short_length",
    "long_length",
    "speed",
    "crs",
    "geometry_field_format",
    "currency",
    "version_number",
    "id_type",
]
NODE_REQUIRED_COLUMNS = ["node_id", "x_coord", "y_coord"]
LINK_REQUIRED_COLUMNS = ["link_id", "from_node_id", "to_node_id", "directed"]


def export_gmns_network(
    gdf_trimmed: gpd.GeoDataFrame,
    *,
    interstate_name: str,
    output_dir: Path,
    projection: str,
    ramps_gdf: gpd.GeoDataFrame | None = None,
) -> Dict[str, Path]:
    """
    Write GMNS node, link, config CSVs and a bundled zip archive.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_df, link_df, config_df = build_gmns_tables(
        gdf_trimmed,
        interstate_name=interstate_name,
        projection=projection,
        ramps_gdf=ramps_gdf,
    )

    node_path = output_dir / "node.csv"
    link_path = output_dir / "link.csv"
    config_path = output_dir / "config.csv"
    archive_path = output_dir.parent / "gmns.zip"

    node_df.to_csv(node_path, index=False)
    link_df.to_csv(link_path, index=False)
    config_df.to_csv(config_path, index=False)
    _write_archive(archive_path, [node_path, link_path, config_path])

    return {
        "node_csv": node_path,
        "link_csv": link_path,
        "config_csv": config_path,
        "archive": archive_path,
    }


def build_gmns_tables(
    gdf_trimmed: gpd.GeoDataFrame,
    *,
    interstate_name: str,
    projection: str,
    ramps_gdf: gpd.GeoDataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert the clipped corridor GeoDataFrame into GMNS tables.
    """
    if gdf_trimmed is None or gdf_trimmed.empty:
        return _prune_empty_columns(_empty_node_df(), NODE_REQUIRED_COLUMNS), _prune_empty_columns(
            _empty_link_df(), LINK_REQUIRED_COLUMNS
        ), _prune_empty_columns(_config_df(interstate_name), [])

    trimmed_wgs84 = gdf_trimmed.to_crs(WGS84).reset_index(drop=True).copy()
    trimmed_projected = trimmed_wgs84.to_crs(projection)
    ramp_records = _build_ramp_records(ramps_gdf, projection)

    node_index: Dict[Tuple[float, float], int] = {}
    node_records_by_id: Dict[int, dict] = {}
    link_records = []

    def register_node(coord: Tuple[float, float], labels: Sequence[str] | None = None) -> int:
        key = _coord_key(coord)
        existing_id = node_index.get(key)
        node_type = _merge_node_type(labels)
        if existing_id is not None:
            if node_type:
                node_records_by_id[existing_id]["node_type"] = _combine_node_types(
                    node_records_by_id[existing_id].get("node_type"),
                    node_type,
                )
            return existing_id

        node_id = len(node_index) + 1
        node_index[key] = node_id
        node_records_by_id[node_id] = {
            "node_id": node_id,
            "name": None,
            "x_coord": float(coord[0]),
            "y_coord": float(coord[1]),
            "z_coord": None,
            "node_type": node_type,
            "ctrl_type": None,
            "zone_id": None,
            "parent_node_id": None,
        }
        return node_id

    link_id = 1
    for row_wgs84, row_projected in zip(
        trimmed_wgs84.itertuples(index=False),
        trimmed_projected.itertuples(index=False),
    ):
        geometry_projected = row_projected.geometry
        if geometry_projected is None or geometry_projected.is_empty:
            continue
        if geometry_projected.length <= 0:
            continue

        marks = _segment_marks(geometry_projected, ramp_records)
        for start_mark, end_mark in zip(marks[:-1], marks[1:]):
            start_distance, start_labels = start_mark
            end_distance, end_labels = end_mark
            if end_distance - start_distance <= 1e-6:
                continue

            split_projected = substring(geometry_projected, start_distance, end_distance)
            if split_projected.is_empty or len(split_projected.coords) < 2:
                continue
            split_wgs84 = _to_wgs84(split_projected, projection)
            if split_wgs84.is_empty or len(split_wgs84.coords) < 2:
                continue

            from_coord = _point_coord(_to_wgs84(geometry_projected.interpolate(start_distance), projection))
            to_coord = _point_coord(_to_wgs84(geometry_projected.interpolate(end_distance), projection))
            from_node_id = register_node(from_coord, start_labels)
            to_node_id = register_node(to_coord, end_labels)

            link_records.append(
                {
                    "link_id": link_id,
                    "name": row_wgs84.name,
                    "from_node_id": from_node_id,
                    "to_node_id": to_node_id,
                    "directed": True,
                    "geometry": split_wgs84.wkt,
                    "dir_flag": 1,
                    "length": float(split_projected.length) / 5280.0,
                    "facility_type": _facility_type(getattr(row_wgs84, "highway", None)),
                    "free_speed": _parse_speed_mph(getattr(row_wgs84, "maxspeed", None)),
                    "lanes": _safe_int(getattr(row_wgs84, "lanes", None)),
                    "osm_way_id": _safe_int(getattr(row_wgs84, "way_id", None)),
                    "ref": getattr(row_wgs84, "ref", None),
                }
            )
            link_id += 1

    node_records = [node_records_by_id[node_id] for node_id in sorted(node_records_by_id)]
    node_df = _prune_empty_columns(pd.DataFrame(node_records, columns=NODE_COLUMNS), NODE_REQUIRED_COLUMNS)
    link_df = _prune_empty_columns(pd.DataFrame(link_records, columns=LINK_COLUMNS), LINK_REQUIRED_COLUMNS)
    config_df = _prune_empty_columns(_config_df(interstate_name), [])

    return node_df, link_df, config_df


def _build_ramp_records(ramps_gdf: gpd.GeoDataFrame | None, projection: str) -> List[dict]:
    if ramps_gdf is None or ramps_gdf.empty:
        return []

    ramps_projected = ramps_gdf.to_crs(projection).copy()
    records = []
    for row in ramps_projected.itertuples(index=False):
        labels = []
        if bool(getattr(row, "entry_node", False)):
            labels.append("on_ramp")
        if bool(getattr(row, "exit_node", False)):
            labels.append("off_ramp")
        if not labels:
            continue
        records.append({"point": row.geometry, "labels": labels})
    return records


def _segment_marks(
    geometry_projected: LineString,
    ramp_records: Sequence[dict],
    tolerance_ft: float = 5.0,
) -> List[Tuple[float, List[str]]]:
    marks: Dict[float, set[str]] = {
        0.0: set(),
        round(float(geometry_projected.length), 6): set(),
    }
    for ramp in ramp_records:
        point = ramp["point"]
        if point is None or point.is_empty:
            continue
        if geometry_projected.distance(point) > tolerance_ft:
            continue
        distance = round(float(geometry_projected.project(point)), 6)
        marks.setdefault(distance, set()).update(ramp["labels"])

    return [(distance, sorted(labels)) for distance, labels in sorted(marks.items())]


def _merge_node_type(labels: Sequence[str] | None) -> str | None:
    if not labels:
        return None
    label_set = set(labels)
    if label_set == {"on_ramp"}:
        return "on_ramp"
    if label_set == {"off_ramp"}:
        return "off_ramp"
    if label_set == {"on_ramp", "off_ramp"}:
        return "on_off_ramp"
    return ",".join(sorted(label_set))


def _combine_node_types(existing: str | None, incoming: str | None) -> str | None:
    if not incoming:
        return existing
    if not existing:
        return incoming
    label_set = set(existing.split(",")) | set(incoming.split(","))
    return _merge_node_type(sorted(label_set))


def _coord_key(coord: Iterable[float], decimals: int = 9) -> Tuple[float, float]:
    x_coord, y_coord = coord
    return (round(float(x_coord), decimals), round(float(y_coord), decimals))


def _point_coord(point: Point) -> Tuple[float, float]:
    return (float(point.x), float(point.y))


def _to_wgs84(geometry, projection: str):
    transformer = Transformer.from_crs(projection, WGS84, always_xy=True)
    return shapely_transform(transformer.transform, geometry)


def _facility_type(highway: object) -> str | None:
    if highway is None:
        return None
    highway_text = str(highway).strip().lower()
    if highway_text == "motorway":
        return "freeway"
    if highway_text == "motorway_link":
        return "ramp"
    return highway_text or None


def _parse_speed_mph(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    speed = float(match.group(0))
    if "km" in text and "mph" not in text:
        return speed * 0.621371
    return speed


def _safe_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prune_empty_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    keep_columns = []
    required_set = set(required_columns)
    for column in df.columns:
        series = df[column]
        if column in required_set:
            keep_columns.append(column)
            continue
        if series.empty:
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            cleaned = series.dropna().astype(str).str.strip()
            if not cleaned.empty and (cleaned != "").any():
                keep_columns.append(column)
        elif series.notna().any():
            keep_columns.append(column)
    return df.loc[:, keep_columns]


def _empty_node_df() -> pd.DataFrame:
    return pd.DataFrame(columns=NODE_COLUMNS)


def _empty_link_df() -> pd.DataFrame:
    return pd.DataFrame(columns=LINK_COLUMNS)


def _config_df(interstate_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": f"{interstate_name} extracted corridor network",
                "short_length": "ft",
                "long_length": "mi",
                "speed": "mph",
                "crs": WGS84,
                "geometry_field_format": "WKT",
                "currency": None,
                "version_number": GMNS_VERSION,
                "id_type": "integer",
            }
        ],
        columns=CONFIG_COLUMNS,
    )


def _write_archive(archive_path: Path, members: Sequence[Path]) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.write(member, arcname=member.name)
