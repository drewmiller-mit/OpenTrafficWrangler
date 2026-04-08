"""
Graph construction and traversal helpers.
"""
from __future__ import annotations

import itertools
import math
import re
import time
from typing import List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import overpy
from shapely.geometry import LineString


def normalize_interstate_name(name: str) -> str:
    """
    Normalize roadway identifiers to an OSM `ref` regex.

    Supported input formats:
    - Interstate: "I 24", "I-24", "I24"
    - State route: "CA SR 55", "CA-55", "CA55"
    """
    cleaned = name.strip().upper()
    compact = re.sub(r"[\s_-]+", "", cleaned)

    interstate_match = re.match(r"I(\d+)", compact)
    if interstate_match:
        digits = interstate_match.group(1)
        return rf"(^|[^0-9A-Z])I[ -]?{digits}($|[^0-9])"

    state_route_match = re.match(r"([A-Z]{2})(?:SR)?(\d+)", compact)
    if state_route_match:
        state, digits = state_route_match.groups()
        return rf"(^|[^0-9A-Z]){state}[ -]?{digits}($|[^0-9])"

    raise ValueError(
        "Route name must be an interstate (e.g., 'I-24') or state route "
        "(e.g., 'CA SR 55')."
    )


def build_mainline_graph(result: overpy.Result, interstate_regex: str | None = None) -> nx.DiGraph:
    """
    Build a directed graph from an Overpass result.

    Edges track the associated OSM way ID and `ref`.
    """
    G = nx.DiGraph()
    for way in result.ways:
        ref = way.tags.get("ref", "")
        highway = way.tags.get("highway", "")
        direction = _way_directionality(way)
        for n1, n2 in zip(way.nodes[:-1], way.nodes[1:]):
            edge_attrs = {
                "way_id": int(way.id),
                "ref": ref,
                "highway": highway,
                "length_m": _haversine_m(float(n1.lat), float(n1.lon), float(n2.lat), float(n2.lon)),
            }
            if direction in {"forward", "both"}:
                G.add_edge(int(n1.id), int(n2.id), **edge_attrs)
            if direction in {"reverse", "both"}:
                G.add_edge(int(n2.id), int(n1.id), **edge_attrs)
    return G


def filtered_subgraph(
    G: nx.DiGraph,
    *,
    highway_values: Optional[set[str]] = None,
    ref_list: Optional[List[str]] = None,
    ref_match_mode: str = "exact",
    require_ref_match: bool = False,
) -> nx.DiGraph:
    """
    Return an edge-induced subgraph with optional highway/ref filtering.
    """
    ref_filters = [p.strip() for p in ref_list or [] if p and p.strip()]

    candidate_edges = []
    for u, v, data in G.edges(data=True):
        highway = (data.get("highway") or "").strip()
        if highway_values is not None and highway not in highway_values:
            continue
        if require_ref_match and not edge_matches_ref(data, ref_filters, ref_match_mode):
            continue
        candidate_edges.append((u, v))

    return G.edge_subgraph(candidate_edges).copy()


def separate_path(
    G: nx.DiGraph,
    start_node: int,
    end_node: int,
    ref_list: Optional[List[str]] = None,
    mode: Optional[str] = None,
    ref_match_mode: str = "exact",
    return_details: bool = False,
    max_candidates: Optional[int] = None,
) -> Tuple[List[int], List[int]] | Tuple[List[int], List[int], dict]:
    """
    Compute a path while optionally preferring/avoiding refs.

    In prefer/avoid mode, candidate simple paths are scored by the prevalence
    of matching refs across path edges. Ties are broken by shorter geometric
    distance.
    """
    if ref_match_mode not in {"exact", "regex"}:
        raise ValueError("ref_match_mode must be 'exact' or 'regex'.")

    ref_filters = [p.strip() for p in ref_list or [] if p and p.strip()]
    start_time = time.perf_counter()
    print(
        f"[separate_path] mode={mode or 'normal'} ref_match_mode={ref_match_mode} "
        f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
        f"start={start_node} end={end_node} ref_filters={ref_filters or '[]'} "
        f"max_candidates={max_candidates if max_candidates is not None else 'unbounded'}"
    )

    def _path_to_way_ids(path_nodes: List[int]) -> Tuple[List[int], List[int]]:
        edges_in_path = []
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            edges_in_path.append(G[u][v]["way_id"])

        seen = set()
        ordered = []
        for wid in edges_in_path:
            if wid not in seen:
                ordered.append(wid)
                seen.add(wid)
        return ordered, path_nodes

    def _path_metrics(path_nodes: List[int]) -> Tuple[int, int, float]:
        matched_edges = 0
        total_edges = 0
        total_length_m = 0.0
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            data = G[u][v]
            total_edges += 1
            total_length_m += float(data.get("length_m") or 0.0)
            if edge_matches_ref(data, ref_filters, ref_match_mode):
                matched_edges += 1
        return matched_edges, total_edges, total_length_m

    def _build_result(path_nodes: List[int], *, details: dict):
        way_ids, nodes = _path_to_way_ids(path_nodes)
        if return_details:
            return way_ids, nodes, details
        return way_ids, nodes

    if ref_filters and mode in {"prefer", "avoid"}:
        try:
            candidates = nx.shortest_simple_paths(G, source=start_node, target=end_node, weight="length_m")
            if max_candidates is not None:
                candidates = itertools.islice(candidates, max_candidates)
            best_nodes = None
            best_matched = -1
            best_total = 1
            best_length_m = math.inf
            total_candidates = 0
            perfect_candidate_count = 0

            for candidate_nodes in candidates:
                total_candidates += 1
                matched_edges, total_edges, total_length_m = _path_metrics(candidate_nodes)
                if total_edges == 0:
                    continue

                better = False
                if best_nodes is None:
                    better = True
                else:
                    current_ratio = matched_edges * best_total
                    best_ratio = best_matched * total_edges
                    if mode == "prefer" and current_ratio > best_ratio:
                        better = True
                    elif mode == "avoid" and current_ratio < best_ratio:
                        better = True
                    elif current_ratio == best_ratio and total_length_m < best_length_m:
                        better = True

                if better:
                    best_nodes = candidate_nodes
                    best_matched = matched_edges
                    best_total = total_edges
                    best_length_m = total_length_m
                    print(
                        f"[separate_path] candidate={total_candidates} new_best "
                        f"matched_edges={best_matched}/{best_total} length_m={best_length_m:.1f}"
                    )

                if total_candidates == 1 or total_candidates % 25 == 0:
                    print(
                        f"[separate_path] scanned_candidates={total_candidates} "
                        f"current_candidate={matched_edges}/{total_edges} length_m={total_length_m:.1f}"
                    )

                is_perfect = (mode == "prefer" and matched_edges == total_edges) or (
                    mode == "avoid" and matched_edges == 0
                )
                if is_perfect:
                    perfect_candidate_count += 1
                if perfect_candidate_count > 1:
                    print(
                        f"[separate_path] multiple perfect candidates found after "
                        f"{total_candidates} candidates; shortest geometric path retained."
                    )
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            raise RuntimeError(f"Failed to compute shortest path: {exc}")

        if best_nodes is None:
            raise RuntimeError(f"Failed to compute shortest path: No path between {start_node} and {end_node}.")

        details = {
            "multiple_candidates": total_candidates > 1,
            "matched_edges": best_matched,
            "total_edges": best_total,
            "length_m": best_length_m,
            "perfect_candidate_count": perfect_candidate_count,
            "used_mode": mode,
            "candidate_limit_hit": max_candidates is not None and total_candidates >= max_candidates,
        }
        elapsed = time.perf_counter() - start_time
        print(
            f"[separate_path] selected mode={mode} matched_edges={best_matched}/{best_total} "
            f"length_m={best_length_m:.1f} scanned_candidates={total_candidates} "
            f"candidate_limit_hit={details['candidate_limit_hit']} elapsed_sec={elapsed:.2f}"
        )
        return _build_result(best_nodes, details=details)

    try:
        candidates = nx.shortest_simple_paths(G, source=start_node, target=end_node, weight="length_m")
        fallback_nodes = next(candidates)
        try:
            next(candidates)
            multiple_candidates = True
        except StopIteration:
            multiple_candidates = False
    except Exception as exc:  # pragma: no cover - bubble up
        raise RuntimeError(f"Failed to compute shortest path: {exc}")

    matched_edges, total_edges, total_length_m = _path_metrics(fallback_nodes)
    if ref_filters:
        non_pref = 0
        avoided = 0
        for u, v in zip(fallback_nodes[:-1], fallback_nodes[1:]):
            if mode == "prefer" and not edge_matches_ref(G[u][v], ref_filters, ref_match_mode):
                non_pref += 1
            if mode == "avoid" and edge_matches_ref(G[u][v], ref_filters, ref_match_mode):
                avoided += 1
        if mode == "prefer" and non_pref:
            print(f"Path includes {non_pref} non-preferred ways (fallback path).")
        if mode == "avoid" and avoided:
            print(f"Path includes {avoided} ways using avoided refs (fallback path).")

    details = {
        "multiple_candidates": multiple_candidates,
        "matched_edges": matched_edges,
        "total_edges": total_edges,
        "length_m": total_length_m,
        "perfect_candidate_count": int(total_edges > 0 and ((mode == "prefer" and matched_edges == total_edges) or (mode == "avoid" and matched_edges == 0))),
        "used_mode": mode or "normal",
        "candidate_limit_hit": False,
    }
    elapsed = time.perf_counter() - start_time
    print(
        f"[separate_path] selected mode={mode or 'normal'} matched_edges={matched_edges}/{total_edges} "
        f"length_m={total_length_m:.1f} multiple_candidates={multiple_candidates} elapsed_sec={elapsed:.2f}"
    )
    return _build_result(fallback_nodes, details=details)


def ways_by_ids_to_gdf(result: overpy.Result, way_ids: List[int]) -> gpd.GeoDataFrame:
    """
    Extract just the requested ways into a GeoDataFrame.
    """
    way_index = {int(w.id): w for w in result.ways}
    records = []
    for wid in way_ids:
        way = way_index.get(int(wid))
        if way is None:
            print(f"[WARN] Way {wid} not found in Overpass result.")
            continue

        coords = [(float(n.lon), float(n.lat)) for n in way.nodes]

        records.append(
            {
                "way_id": int(way.id),
                "name": way.tags.get("name"),
                "ref": way.tags.get("ref"),
                "highway": way.tags.get("highway"),
                "lanes": _safe_int(way.tags.get("lanes")),
                "maxspeed": way.tags.get("maxspeed"),
                "geometry": LineString(coords),
                "hov": way.tags.get("hov"),
            }
        )

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def _safe_int(val):
    try:
        return int(val) if val is not None and val != "" else None
    except Exception:
        return None


def edge_matches_ref(data: dict, ref_list: Optional[List[str]], ref_match_mode: str = "exact") -> bool:
    ref_filters = [p.strip() for p in ref_list or [] if p and p.strip()]
    if not ref_filters:
        return False
    ref = (data.get("ref") or "").strip()
    if ref_match_mode == "exact":
        return ref in set(ref_filters)
    if ref_match_mode == "regex":
        return any(re.search(pattern, ref) for pattern in ref_filters)
    raise ValueError("ref_match_mode must be 'exact' or 'regex'.")


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_m * c


def _way_directionality(way: overpy.Way) -> str:
    tags = way.tags or {}
    oneway = (tags.get("oneway", "") or "").strip().lower()

    if oneway in {"yes", "true", "1"}:
        return "forward"
    if oneway == "-1":
        return "reverse"
    if oneway in {"no", "false", "0"}:
        return "both"
    return "both"
