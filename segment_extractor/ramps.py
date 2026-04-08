"""
Ramp/node classification utilities.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

import geopandas as gpd
import pandas as pd
import overpy
from shapely.geometry import Point


def _safe_int(value):
    try:
        return int(value) if value not in (None, "") else None
    except Exception:
        return None


def _is_excluded_connecting_way(way: overpy.Way) -> bool:
    tags = way.tags or {}
    highway = (tags.get("highway", "") or "").lower()
    if not highway:
        return True
    if (tags.get("access", "") or "").lower() == "private":
        return True
    if highway == "service":
        return True
    if highway == "proposed":
        return True
    return False


def process_ramp_data(
    segment_data: overpy.Result,
    path_nodes: List[int],
    mainline_way_ids: List[int],
    projected_crs: str,
):
    """
    Convert Overpass segment data into nodes_df using entry/exit classification.
    """
    path_node_set = set(map(int, path_nodes))
    mainline_way_id_set = set(map(int, mainline_way_ids))

    node_to_ways = defaultdict(list)
    for way in segment_data.ways:
        for node in way.nodes:
            node_to_ways[int(node.id)].append(way)

    entry_nodes = set()
    exit_nodes = set()
    entry_node_lanes = defaultdict(list)

    for node_id in path_node_set:
        connected_ways = node_to_ways.get(node_id, [])
        if not connected_ways:
            continue

        mainline_ways = []
        connecting_ways = []
        for way in connected_ways:
            if int(way.id) in mainline_way_id_set:
                mainline_ways.append(way)
            else:
                if _is_excluded_connecting_way(way):
                    continue
                connecting_ways.append(way)

        n_main = len(mainline_ways)
        if n_main == 0 or not connecting_ways:
            continue

        for ramp_way in connecting_ways:
            if _is_excluded_connecting_way(ramp_way):
                continue
            ramp_nodes = [int(node.id) for node in ramp_way.nodes]
            if len(ramp_nodes) < 2:
                continue

            if node_id == ramp_nodes[0]:
                exit_nodes.add(node_id)
            elif node_id == ramp_nodes[-1]:
                entry_nodes.add(node_id)
                ramp_lanes = _safe_int((ramp_way.tags or {}).get("lanes"))
                if ramp_lanes is not None:
                    entry_node_lanes[node_id].append(ramp_lanes)

    print(f"Detected {len(entry_nodes)} entry nodes and {len(exit_nodes)} exit nodes (>=1 connecting ways).")

    node_records = []
    geometries = []
    for node in segment_data.nodes:
        nid = int(node.id)
        if nid not in path_node_set:
            continue
        node_records.append(
            {
                "node_id": nid,
                "lat": float(node.lat),
                "lon": float(node.lon),
                "entry_node": nid in entry_nodes,
                "exit_node": nid in exit_nodes,
                "num_lanes": (
                    max(entry_node_lanes[nid]) if nid in entry_nodes and entry_node_lanes.get(nid) else pd.NA
                ),
            }
        )
        geometries.append(Point(float(node.lon), float(node.lat)))

    nodes_gdf = gpd.GeoDataFrame(node_records, geometry=geometries, crs="EPSG:4326")
    nodes_gdf = nodes_gdf[(nodes_gdf["entry_node"] == True) | (nodes_gdf["exit_node"] == True)]
    print(f"Built nodes_df with {len(nodes_gdf)} mainline ramp nodes")
    return nodes_gdf
