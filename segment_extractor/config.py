"""
Configuration helpers and constants for the segment extractor.
"""
from __future__ import annotations

import geopandas as gpd
from pyproj import network
from shapely.geometry import Point


network.set_network_enabled(False)

STATEPLANE_FT = {
    "AL": "EPSG:26929",
    "AK": "EPSG:3338",
    "AZ": "EPSG:2223",
    "AR": "EPSG:26951",
    "CA": "EPSG:2227",
    "CO": "EPSG:2232",
    "CT": "EPSG:26956",
    "DE": "EPSG:26957",
    "FL": "EPSG:2236",
    "GA": "EPSG:2240",
    "HI": "EPSG:3759",
    "ID": "EPSG:2241",
    "IL": "EPSG:26971",
    "IN": "EPSG:26973",
    "IA": "EPSG:26975",
    "KS": "EPSG:26977",
    "KY": "EPSG:2246",
    "LA": "EPSG:26981",
    "ME": "EPSG:26986",
    "MD": "EPSG:26985",
    "MA": "EPSG:26986",
    "MI": "EPSG:2253",
    "MN": "EPSG:26992",
    "MS": "EPSG:2256",
    "MO": "EPSG:26994",
    "MT": "EPSG:32100",
    "NE": "EPSG:32104",
    "NV": "EPSG:2258",
    "NH": "EPSG:26989",
    "NJ": "EPSG:3424",
    "NM": "EPSG:2259",
    "NY": "EPSG:2260",
    "NC": "EPSG:2264",
    "ND": "EPSG:2265",
    "OH": "EPSG:3734",
    "OK": "EPSG:2269",
    "OR": "EPSG:2992",
    "PA": "EPSG:2272",
    "RI": "EPSG:32130",
    "SC": "EPSG:2273",
    "SD": "EPSG:32134",
    "TN": "EPSG:2274",
    "TX": "EPSG:2277",
    "UT": "EPSG:3566",
    "VT": "EPSG:32145",
    "VA": "EPSG:2284",
    "WA": "EPSG:2286",
    "WV": "EPSG:32153",
    "WI": "EPSG:3071",
    "WY": "EPSG:32155",
}

# Load USA states (lower 48 + AK/HI) ONCE
US_STATES = gpd.read_file(
    "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
)
US_STATES["state_abbr"] = US_STATES["STUSPS"]


def projection_from_segment_coordinates(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> str:
    """
    Determine the proper StatePlane EPSG (US feet) by checking which state
    contains the midpoint of the requested segment.
    """
    mid_lat = (start_lat + end_lat) / 2
    mid_lon = (start_lon + end_lon) / 2
    pt = Point(mid_lon, mid_lat)

    match = US_STATES[US_STATES.contains(pt)]
    if len(match) == 0:
        raise ValueError("Coordinates are not inside the United States.")

    state_abbr = match.iloc[0]["state_abbr"]
    epsg = STATEPLANE_FT.get(state_abbr)
    if epsg is None:
        raise ValueError(f"No StatePlane EPSG found for state {state_abbr}")

    return epsg
