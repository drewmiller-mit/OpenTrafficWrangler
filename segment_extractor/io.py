"""
Network/Overpass access utilities.
"""
from __future__ import annotations

import time
from typing import Callable, List

import overpy


def _run_with_retry(func: Callable[[], overpy.Result], *, timeout_sec: float = 60.0, delay: float = 5.0):
    """
    Execute `func` repeatedly until it succeeds or the timeout elapses.
    """
    deadline = time.monotonic() + timeout_sec
    attempt = 0
    last_exc = None
    while time.monotonic() <= deadline:
        attempt += 1
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            remaining = max(0.0, deadline - time.monotonic())
            print(f"[Overpass] attempt {attempt} failed ({exc}); retrying in {delay}s ({remaining:.1f}s left)")
            time.sleep(delay)
    raise RuntimeError(f"Overpass query failed after {attempt} attempts: {last_exc}")


def overpass_query_interstate_bbox(
    interstate_regex: str,
    seg_start_lat: float,
    seg_start_lon: float,
    seg_end_lat: float,
    seg_end_lon: float,
    buffer: float,
    timeout: int = 360,
) -> overpy.Result:
    """
    Query Overpass API for motorway/motorway_link ways matching the route regex within a bbox.
    """
    print("interstate regex", interstate_regex)
    api = overpy.Overpass()

    min_lat = min(seg_start_lat, seg_end_lat) - buffer
    max_lat = max(seg_start_lat, seg_end_lat) + buffer
    min_lon = min(seg_start_lon, seg_end_lon) - buffer
    max_lon = max(seg_start_lon, seg_end_lon) + buffer

    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["highway"~"^(motorway|motorway_link)$"]["ref"~"{interstate_regex}"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    (._;>;);
    out body;
    """
    return _run_with_retry(lambda: api.query(query))


def overpass_query_snapping_bbox(
    seg_start_lat: float,
    seg_start_lon: float,
    seg_end_lat: float,
    seg_end_lon: float,
    buffer: float,
    timeout: int = 360,
) -> overpy.Result:
    """
    Query Overpass API for all highway-tagged ways within a bbox.
    """
    api = overpy.Overpass()

    min_lat = min(seg_start_lat, seg_end_lat) - buffer
    max_lat = max(seg_start_lat, seg_end_lat) + buffer
    min_lon = min(seg_start_lon, seg_end_lon) - buffer
    max_lon = max(seg_start_lon, seg_end_lon) + buffer

    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    (._;>;);
    out body;
    """
    return _run_with_retry(lambda: api.query(query))


def overpass_query_motorway_bbox(
    seg_start_lat: float,
    seg_start_lon: float,
    seg_end_lat: float,
    seg_end_lon: float,
    buffer: float,
    timeout: int = 360,
) -> overpy.Result:
    """
    Query Overpass API for all motorway/motorway_link ways within a bbox.
    """
    api = overpy.Overpass()

    min_lat = min(seg_start_lat, seg_end_lat) - buffer
    max_lat = max(seg_start_lat, seg_end_lat) + buffer
    min_lon = min(seg_start_lon, seg_end_lon) - buffer
    max_lon = max(seg_start_lon, seg_end_lon) + buffer

    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["highway"~"^(motorway|motorway_link)$"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    (._;>;);
    out body;
    """
    return _run_with_retry(lambda: api.query(query))


def overpass_query_highway_bbox(
    seg_start_lat: float,
    seg_start_lon: float,
    seg_end_lat: float,
    seg_end_lon: float,
    buffer: float,
    timeout: int = 360,
) -> overpy.Result:
    """
    Query Overpass API for motorway, motorway_link, and primary ways within a bbox.
    """
    api = overpy.Overpass()

    min_lat = min(seg_start_lat, seg_end_lat) - buffer
    max_lat = max(seg_start_lat, seg_end_lat) + buffer
    min_lon = min(seg_start_lon, seg_end_lon) - buffer
    max_lon = max(seg_start_lon, seg_end_lon) + buffer

    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["highway"~"^(motorway|motorway_link|primary)$"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    (._;>;);
    out body;
    """
    return _run_with_retry(lambda: api.query(query))


def overpy_result_to_json(result):
    """
    Convert Overpass results to a JSON-like dict for inspection.
    """
    elements = []

    for n in result.nodes:
        elements.append(
            {
                "type": "node",
                "id": n.id,
                "lat": float(n.lat),
                "lon": float(n.lon),
                "tags": n.tags,
            }
        )

    for w in result.ways:
        elements.append(
            {
                "type": "way",
                "id": w.id,
                "nodes": [int(nd.id) for nd in w.nodes],
                "tags": w.tags,
            }
        )

    for r in result.relations:
        elements.append(
            {
                "type": "relation",
                "id": r.id,
                "members": [
                    {
                        "type": m._type_value,
                        "ref": int(m.ref),
                        "role": m.role,
                    }
                    for m in r.members
                ],
                "tags": r.tags,
            }
        )

    return {"elements": elements}


def fetch_corridor_segment_with_ramps(
    way_ids: List[int], timeout: int = 360
) -> overpy.Result:
    """
    Fetch mainline ways plus any connected ways/nodes.
    """
    api = overpy.Overpass()
    way_ids = list(dict.fromkeys(way_ids))
    print(f"Fetching {len(way_ids)} ways for corridor...")

    ids_str = ",".join(map(str, way_ids))
    query = f"""
    [out:json][timeout:{timeout}];

    way(id:{ids_str})->.main;
    node(w.main)->.main_nodes;
    way(bn.main_nodes)->.connected_ways;
    (.main; .connected_ways;)->.all_ways;
    node(w.all_ways)->.all_nodes;
    (.all_ways; .all_nodes;);
    out body geom;
    """
    print("Querying Overpass API for corridor + all connected ways...")
    result = _run_with_retry(lambda: api.query(query))
    print(f"Downloaded {len(result.ways)} ways and {len(result.nodes)} nodes")
    return result
