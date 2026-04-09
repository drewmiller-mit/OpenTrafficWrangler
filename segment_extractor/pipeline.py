"""
High-level orchestration for the interstate segment extraction workflow.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import networkx as nx
from .config import projection_from_segment_coordinates
from .geometry import (
    aggregate_lane_blocks,
    clip_line_between_coords,
    compute_point_stationing,
    compute_segment_stationing,
    merge_segments_to_line,
)
from .gmns import export_gmns_network
from shapely.geometry import LineString, Point

from .graph import build_mainline_graph, filtered_subgraph, normalize_interstate_name, separate_path, ways_by_ids_to_gdf
from .io import (
    overpass_query_highway_bbox,
    overpy_result_to_json,
)
from .ramps import process_ramp_data
from .visualization import debug_plot_all_ways, debug_plot_gdf


def _nearest_node_id(nodes_lookup, candidate_node_ids, target_coord, preferred_opposite_vector=None):
    """
    Find the OSM node closest to the provided (lon, lat) coordinate.
    If preferred_opposite_vector is provided, bias toward nodes that lie
    in the opposite direction of that vector (dot product <= 0).
    """
    target_lon, target_lat = target_coord
    best_node = None
    best_dist = float("inf")
    fallback_node = None
    fallback_dist = float("inf")

    for node_id in candidate_node_ids:
        node = nodes_lookup.get(int(node_id))
        if node is None:
            continue
        lon = float(node.lon)
        lat = float(node.lat)
        dist = (lon - target_lon) ** 2 + (lat - target_lat) ** 2
        if preferred_opposite_vector:
            dot = (lon - target_lon) * preferred_opposite_vector[0] + (lat - target_lat) * preferred_opposite_vector[1]
            if dot <= 0 and dist < best_dist:
                best_dist = dist
                best_node = int(node.id)
        if dist < fallback_dist:
            fallback_dist = dist
            fallback_node = int(node.id)

    if best_node is not None:
        return best_node
    if fallback_node is None:
        raise ValueError("Unable to infer OSM node from the provided coordinate.")
    return fallback_node


def _nodes_on_nearest_way(result, coord):
    """
    Identify the ordered node IDs that make up the way closest to `coord`.
    """
    nearest_way = _nearest_way(result, coord)
    return [int(n.id) for n in nearest_way.nodes]


def _nearest_way(result, coord):
    """
    Identify the OSM way closest to `coord`.
    """
    target = Point(coord)
    closest_way = None
    best_dist = float("inf")
    for way in result.ways:
        coords = [(float(n.lon), float(n.lat)) for n in way.nodes]
        if len(coords) < 2:
            continue
        line = LineString(coords)
        dist = line.distance(target)
        if dist < best_dist:
            best_dist = dist
            closest_way = way
    if closest_way is None:
        raise ValueError("Unable to locate a nearby roadway segment for snapping.")
    return closest_way


def _infer_path_nodes(result, start_coord, end_coord):
    vector_start_to_end = (end_coord[0] - start_coord[0], end_coord[1] - start_coord[1])
    vector_end_to_start = (start_coord[0] - end_coord[0], start_coord[1] - end_coord[1])
    nodes_lookup = {int(node.id): node for node in result.nodes}
    start_way = _nearest_way(result, start_coord)
    end_way = _nearest_way(result, end_coord)
    start_way_nodes = [int(n.id) for n in start_way.nodes]
    end_way_nodes = [int(n.id) for n in end_way.nodes]

    start_node = _nearest_node_id(
        nodes_lookup, start_way_nodes, start_coord, preferred_opposite_vector=vector_start_to_end
    )
    end_node = _nearest_node_id(
        nodes_lookup, end_way_nodes, end_coord, preferred_opposite_vector=vector_end_to_start
    )

    print(
        "[path_infer] start nearest way "
        f"id={int(start_way.id)} highway={start_way.tags.get('highway')} "
        f"ref={start_way.tags.get('ref')} name={start_way.tags.get('name')} "
        f"selected_node={start_node}"
    )
    print(
        "[path_infer] end nearest way "
        f"id={int(end_way.id)} highway={end_way.tags.get('highway')} "
        f"ref={end_way.tags.get('ref')} name={end_way.tags.get('name')} "
        f"selected_node={end_node}"
    )
    print(
        f"Derived start node {start_node} and end node {end_node} from clicked coordinates "
        "(biased to extend beyond the selected span)."
    )
    return start_node, end_node


def run_pipeline(
    interstate_name: str,
    seg_start_lat: float,
    seg_start_lon: float,
    seg_end_lat: float,
    seg_end_lon: float,
    out_lanes_csv: str,
    out_ramps_csv: str,
    anchor_postmile: float,
    end_postmile: float | None,
    bbox_buffer_ft: float,
    path_mode: str,
    ref_list: Optional[List[str]],
    start_osm_node: Optional[int] = None,
    end_osm_node: Optional[int] = None,
    intermediate_dir: Optional[Path] = None,
    stationing_direction: str = "ascending",
    outputs_root: Optional[Path] = None,
    allow_relaxation: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Full orchestration entry point used by CLI/API surfaces.
    """
    if stationing_direction not in {"ascending", "descending"}:
        raise ValueError("stationing_direction must be 'ascending' or 'descending'.")
    outputs_root = (
        Path("outputs")
        if outputs_root is None
        else Path(outputs_root)
    )
    if intermediate_dir is None:
        intermediate_dir = outputs_root / "intermediates"
    else:
        intermediate_dir = Path(intermediate_dir)
    interstate_dir = outputs_root / interstate_name
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    interstate_dir.mkdir(parents=True, exist_ok=True)

    projection = projection_from_segment_coordinates(
        seg_start_lat, seg_start_lon, seg_end_lat, seg_end_lon
    )
    interstate_regex = normalize_interstate_name(interstate_name)
    start_coord = (float(seg_start_lon), float(seg_start_lat))
    end_coord = (float(seg_end_lon), float(seg_end_lat))

    def _emit(message: str) -> None:
        print(message)
        if log_fn is not None:
            log_fn(message)

    def _is_relaxable_path_failure(exc: RuntimeError) -> bool:
        message = str(exc)
        return "No path between" in message or "not in graph" in message

    print(f"[1/7] Querying Overpass for {interstate_name} highway network...")
    result = overpass_query_highway_bbox(
        seg_start_lat,
        seg_start_lon,
        seg_end_lat,
        seg_end_lon,
        bbox_buffer_ft,
    )
    debug_plot_all_ways(result, projection)
    if result.ways:
        snapshot_path = intermediate_dir / f"{interstate_name}_overpass_query.json"
        snapshot_path.write_text(json.dumps(overpy_result_to_json(result), indent=2))
        all_way_ids = [int(w.id) for w in result.ways]
        raw_path = intermediate_dir / f"{interstate_name}_overpass_query.csv"
        gdf_all = ways_by_ids_to_gdf(result, all_way_ids)
        if not gdf_all.empty:
            gdf_all.to_csv(raw_path, index=False)
            debug_plot_gdf(
                gdf_all,
                f"{interstate_name} Overpass query",
                projection,
                f"{interstate_name}_overpass_query",
                out_dir=str(intermediate_dir)
            )

    if start_osm_node is None or end_osm_node is None:
        start_osm_node, end_osm_node = _infer_path_nodes(result, start_coord, end_coord)

    print(f"[2/7] Building graph and finding path from {start_osm_node} to {end_osm_node}...")
    G = build_mainline_graph(result, interstate_regex)
    path_source_result = result
    max_path_candidates = 10
    motorway_graph = filtered_subgraph(
        G,
        highway_values={"motorway", "motorway_link","primary"},
    )
    stage_1_regex_required_graph = filtered_subgraph(
        G,
        highway_values={"motorway", "motorway_link", "primary"},
        ref_list=[interstate_regex],
        ref_match_mode="regex",
        require_ref_match=True,
    )
    regex_required_all_roads_graph = filtered_subgraph(
        G,
        ref_list=[interstate_regex],
        ref_match_mode="regex",
        require_ref_match=True,
    )

    stages = [
        {
            "name": "stage_1_motorway_link_primary_ref_required",
            "graph": stage_1_regex_required_graph,
            "mode": None,
            "ref_list": None,
            "ref_match_mode": "exact",
        },
        {
            "name": "stage_2_motorway_ref_prefer_regex",
            "graph": motorway_graph,
            "mode": "prefer",
            "ref_list": [interstate_regex],
            "ref_match_mode": "regex",
        },
        {
            "name": "stage_3_all_highways_ref_required",
            "graph": regex_required_all_roads_graph,
            "mode": None,
            "ref_list": None,
            "ref_match_mode": "exact",
        },
        {
            "name": "stage_4_full_graph_ref_prefer_regex",
            "graph": G,
            "mode": "prefer",
            "ref_list": [interstate_regex],
            "ref_match_mode": "regex",
        },
    ]

    way_ids = None
    path_nodes = None
    selected_graph = None
    selected_details = None
    used_relaxation = False

    for index, stage in enumerate(stages):
        stage_graph = stage["graph"]
        stage_start = time.perf_counter()
        contains_start = stage_graph.has_node(start_osm_node)
        contains_end = stage_graph.has_node(end_osm_node)
        same_weak_component = None
        reverse_path_exists = None
        if contains_start and contains_end:
            undirected_view = stage_graph.to_undirected(as_view=True)
            same_weak_component = nx.has_path(undirected_view, start_osm_node, end_osm_node)
            reverse_path_exists = nx.has_path(stage_graph, end_osm_node, start_osm_node)
        _emit(
            f"[path_stage] {stage['name']} nodes={stage_graph.number_of_nodes()} edges={stage_graph.number_of_edges()} "
            f"contains_start={contains_start} contains_end={contains_end} "
            f"same_weak_component={same_weak_component} reverse_path_exists={reverse_path_exists} "
            f"mode={stage['mode'] or 'normal'} ref_match_mode={stage['ref_match_mode']}"
        )
        try:
            way_ids, path_nodes, selected_details = separate_path(
                stage_graph,
                start_osm_node,
                end_osm_node,
                stage["ref_list"],
                stage["mode"],
                stage["ref_match_mode"],
                return_details=True,
                max_candidates=max_path_candidates,
            )
            selected_graph = stage_graph
            used_relaxation = index > 0
            _emit(
                f"[path_stage] {stage['name']} succeeded way_count={len(way_ids)} "
                f"path_node_count={len(path_nodes)} elapsed_sec={time.perf_counter() - stage_start:.2f}"
            )
            break
        except RuntimeError as exc:
            _emit(
                f"[path_stage] {stage['name']} failed reason={exc} "
                f"elapsed_sec={time.perf_counter() - stage_start:.2f}"
            )
            if not _is_relaxable_path_failure(exc):
                raise
            if index == len(stages) - 1:
                raise RuntimeError(
                    f"{exc}. Unable to extract a directed mainline path after all relaxation steps."
                )
            _emit("no subgraph found, relaxing constraints")

    exact_mode_requested = path_mode in {"prefer", "avoid"} and bool(ref_list)
    if selected_details and selected_details.get("multiple_candidates"):
        if exact_mode_requested:
            way_ids, path_nodes, exact_details = separate_path(
                selected_graph,
                start_osm_node,
                end_osm_node,
                ref_list,
                path_mode,
                "exact",
                return_details=True,
                max_candidates=max_path_candidates,
            )
            selected_details = exact_details
            if exact_details.get("candidate_limit_hit"):
                _emit("candidate path search capped; exact ref disambiguation evaluated only the shortest candidate set.")
            if exact_details.get("perfect_candidate_count", 0) > 1 or exact_details.get("perfect_candidate_count", 0) == 0:
                _emit("multiple possible paths found--shortest path extracted. please review for correctness.")
        else:
            if selected_details.get("candidate_limit_hit"):
                _emit("candidate path search capped; shortest candidate set evaluated.")
            _emit("multiple possible paths found--shortest path extracted. please review for correctness.")

    print(f"[3/7] Fetching geometry for {len(way_ids)} ways along the path...")
    gdf_path = ways_by_ids_to_gdf(path_source_result, way_ids)
    gdf_path.to_csv(intermediate_dir / f"{interstate_name}_isolated_mainline_path.csv")
    debug_plot_gdf(
        gdf_path,
        f"{interstate_name} isolated mainline path",
        projection,
        f"{interstate_name}_isolated_mainline_path",
        out_dir=str(intermediate_dir)
    )

    print(f"[4/7] Clipping path to bounding box defined by provided segment endpoints...")
    gdf_trimmed = clip_line_between_coords(gdf_path, start_coord, end_coord, projection=projection)
    gdf_trimmed.to_csv(intermediate_dir / f"{interstate_name}_clipped_to_endpoints.csv")
    trimmed_geojson_path = interstate_dir / "clipped_mainline.geojson"
    if gdf_trimmed.empty:
        trimmed_geojson = {"type": "FeatureCollection", "features": []}
    else:
        trimmed_geojson = json.loads(gdf_trimmed.to_crs("EPSG:4326").to_json())
    trimmed_geojson_path.write_text(json.dumps(trimmed_geojson), encoding="utf-8")
    debug_plot_gdf(
        gdf_trimmed,
        f"{interstate_name} clipped to endpoints",
        projection,
        f"{interstate_name}_clipped_to_endpoints",
        show_legend=False,
        figsize=(2, 10),
        out_dir=str(intermediate_dir),
    )
    gdf_trimmed_proj = gdf_trimmed.to_crs(projection) if not gdf_trimmed.empty else gdf_trimmed
    if gdf_trimmed_proj is not None and not gdf_trimmed_proj.empty:
        ref_line_trimmed = merge_segments_to_line(gdf_trimmed_proj)
        segment_length_mi = ref_line_trimmed.length / 5280.0
    else:
        segment_length_mi = 0.0

    print(f"[5/7] Computing RCS for lane segments and aggregating contiguous blocks...")
    # Lane stationing should cover the full extracted corridor, including
    # mainline connectors that are represented as motorway_link in OSM.
    gdf_main = gdf_trimmed[gdf_trimmed["highway"].isin({"motorway", "motorway_link"})].copy()
    lanes_missing = gdf_main["lanes"].isna().sum()
    if lanes_missing > 0:
        print(f"  Note: {lanes_missing} segment(s) missing lane counts; dropped from lane aggregation.")
        gdf_main = gdf_main[~gdf_main["lanes"].isna()].copy()

    anchor_coord = start_coord
    segments_stationed = compute_segment_stationing(
        gdf_main,
        gdf_trimmed,
        anchor_coord=anchor_coord,
        anchor_postmile=anchor_postmile,
        projection=projection,
        stationing_direction=stationing_direction,
        end_postmile=end_postmile,
    )
    segments_stationed.to_csv(intermediate_dir / f"{interstate_name}_stationed_lane_configs.csv")

    lanes_rcs = aggregate_lane_blocks(segments_stationed, max_gap_ft=500.0)
    if lanes_rcs.empty:
        print("  Warning: No lane segments detected; lanes CSV will be empty.")
    lanes_rcs.to_csv(interstate_dir / out_lanes_csv, index=False)
    debug_plot_gdf(
        segments_stationed,
        f"{interstate_name} lane drop/add locations",
        projection,
        f"{interstate_name}_stationed_lane_configs",
        out_dir=str(intermediate_dir)
    )
    print(f"  Wrote lanes CSV: {interstate_dir / out_lanes_csv}")

    print(f"[6/7] Detecting ramp nodes and computing their RCS positions...")
    if used_relaxation:
        print("  Note: Relaxed graph filtering was used for path extraction.")
    ramps_gdf = process_ramp_data(result, path_nodes, way_ids, projection)
    ramps_stationed = compute_point_stationing(
        ramps_gdf,
        gdf_trimmed,
        anchor_coord=anchor_coord,
        anchor_postmile=anchor_postmile,
        projection=projection,
        segment_length_mi=segment_length_mi,
        stationing_direction=stationing_direction,
        end_postmile=end_postmile,
    )
    debug_plot_gdf(
        ramps_stationed,
        f"{interstate_name} on/off ramp locations",
        projection,
        f"{interstate_name}_ramp_nodes",
        out_dir=str(intermediate_dir)
    )
    gmns_paths = export_gmns_network(
        gdf_trimmed,
        interstate_name=interstate_name,
        output_dir=interstate_dir / "gmns",
        projection=projection,
        ramps_gdf=ramps_stationed,
    )
    print(f"  Wrote GMNS node CSV: {gmns_paths['node_csv']}")
    print(f"  Wrote GMNS link CSV: {gmns_paths['link_csv']}")
    print(f"  Wrote GMNS config CSV: {gmns_paths['config_csv']}")
    print(f"  Wrote GMNS archive: {gmns_paths['archive']}")

    out_cols = ["node_id", "geometry", "x_rcs_miles", "entry_node", "exit_node", "num_lanes"]
    ramps_out = ramps_stationed.copy()
    ramps_out = ramps_out[out_cols]
    ramps_out["lon"] = ramps_out.geometry.x
    ramps_out["lat"] = ramps_out.geometry.y
    ramps_out.to_csv(interstate_dir / out_ramps_csv, index=False)

    print(f"  Wrote ramps CSV: {interstate_dir / out_ramps_csv}")
    print("[7/7] Done.")
