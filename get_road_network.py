#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Roadway Segment Extractor CLI entry point.

The heavy lifting now lives under `segment_extractor`. This file
keeps backward compatibility for notebooks/scripts that still import helpers
directly from `unified_get_segment.py`, while exposing the same CLI behavior.
"""
from __future__ import annotations

import argparse

from segment_extractor.config import projection_from_segment_coordinates
from segment_extractor.geometry import (
    aggregate_lane_blocks,
    clip_line_between_coords,
    compute_point_stationing,
    compute_segment_stationing,
    fit_reference_spline,
    merge_segments_to_line,
    to_state_plane,
)
from segment_extractor.graph import (
    build_mainline_graph,
    normalize_interstate_name,
    separate_path,
    ways_by_ids_to_gdf,
)
from segment_extractor.io import (
    fetch_corridor_segment_with_ramps,
    overpass_query_interstate_bbox,
    overpass_query_snapping_bbox,
    overpy_result_to_json,
)
from segment_extractor.pipeline import run_pipeline
from segment_extractor.validation_utils import (
    generate_validation_kml,
    generate_validation_osm_html,
)
from segment_extractor.ramps import process_ramp_data
from segment_extractor.visualization import (
    apply_white_halo_to_axes,
    debug_plot_all_ways,
    debug_plot_gdf,
    plot_all_data,
)

__all__ = [
    "aggregate_lane_blocks",
    "apply_white_halo_to_axes",
    "build_mainline_graph",
    "clip_line_between_coords",
    "compute_point_stationing",
    "compute_segment_stationing",
    "debug_plot_all_ways",
    "debug_plot_gdf",
    "fetch_corridor_segment_with_ramps",
    "fit_reference_spline",
    "merge_segments_to_line",
    "normalize_interstate_name",
    "overpass_query_interstate_bbox",
    "overpass_query_snapping_bbox",
    "overpy_result_to_json",
    "plot_all_data",
    "process_ramp_data",
    "projection_from_segment_coordinates",
    "run_pipeline",
    "separate_path",
    "to_state_plane",
    "ways_by_ids_to_gdf",
]


def main():
    parser = argparse.ArgumentParser(
        description="Extract and station a roadway segment with lanes and ramp nodes."
    )
    parser.add_argument(
        "--interstate",
        required=True,
        help='Route name, e.g., "I-24" or "CA SR 55"',
    )
    parser.add_argument("--preferences", type=list, required=False)
    parser.add_argument(
        "--start_node",
        type=int,
        required=False,
        help="Optional OSM node ID for the start of the corridor (normally inferred).",
    )
    parser.add_argument(
        "--end_node",
        type=int,
        required=False,
        help="Optional OSM node ID for the end of the corridor (normally inferred).",
    )
    parser.add_argument("--seg_start_lat", type=float, required=True, help="Segment start latitude")
    parser.add_argument("--seg_start_lon", type=float, required=True, help="Segment start longitude")
    parser.add_argument("--seg_end_lat", type=float, required=True, help="Segment end latitude")
    parser.add_argument("--seg_end_lon", type=float, required=True, help="Segment end longitude")
    parser.add_argument("--anchor_postmile", default=0, type=float, help="Anchor postmile for RCS stationing")
    parser.add_argument(
        "--end_postmile",
        default=None,
        type=float,
        help="Optional end postmile for two-point linear station calibration.",
    )
    parser.add_argument(
        "--bbox_buffer_ft", default=0.08, type=float, help="Bounding-box buffer in feet (default 0.05)"
    )
    parser.add_argument("--out_lanes_csv", default="lanes.csv", help="Output CSV for lane blocks")
    parser.add_argument("--out_ramps_csv", default="ramps.csv", help="Output CSV for ramp nodes")
    parser.add_argument(
        "--path_mode",
        choices=["normal", "prefer", "avoid"],
        default="normal",
        help="Pathfinding mode: normal, prefer, or avoid certain refs",
    )
    parser.add_argument("--ref_list", nargs="*", default=None, help="List of reference IDs to consider")
    parser.add_argument(
        "--generate_validation",
        choices=["none", "kml", "osm", "both"],
        default="none",
        help="Optionally emit validation artifacts after the run.",
    )
    parser.add_argument(
        "--stationing_direction",
        choices=["ascending", "descending"],
        default="ascending",
        help="Whether postmile stationing increases (ascending) or decreases (descending) downstream.",
    )
    parser.add_argument(
        "--allow_relaxation",
        action="store_true",
        help=(
            "Deprecated; retained for compatibility. The pipeline now relaxes constraints automatically."
        ),
    )
    args = parser.parse_args()

    run_pipeline(
        interstate_name=args.interstate,
        seg_start_lat=args.seg_start_lat,
        seg_start_lon=args.seg_start_lon,
        seg_end_lat=args.seg_end_lat,
        seg_end_lon=args.seg_end_lon,
        out_lanes_csv=args.out_lanes_csv,
        out_ramps_csv=args.out_ramps_csv,
        anchor_postmile=args.anchor_postmile,
        end_postmile=args.end_postmile,
        bbox_buffer_ft=args.bbox_buffer_ft,
        path_mode=args.path_mode,
        ref_list=args.ref_list,
        start_osm_node=args.start_node,
        end_osm_node=args.end_node,
        stationing_direction=args.stationing_direction,
        allow_relaxation=args.allow_relaxation,
    )

    if args.generate_validation in {"kml", "both"}:
        generate_validation_kml(args.interstate, args.out_lanes_csv, args.out_ramps_csv)
    if args.generate_validation in {"osm", "both"}:
        generate_validation_osm_html(args.interstate, args.out_lanes_csv, args.out_ramps_csv)


if __name__ == "__main__":
    main()
