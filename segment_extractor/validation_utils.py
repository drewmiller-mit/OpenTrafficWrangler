"""Shared utilities for generating validation artifacts (KML, OSM HTML)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from shapely import wkt
from shapely.geometry import Point, mapping, shape
from simplekml import Color, Kml


OUTPUT_ROOT = Path("outputs")


def _prepare_lat_lon(df: pd.DataFrame, lat_candidates: List[str], lon_candidates: List[str]) -> pd.DataFrame:
    lat_col = next((col for col in lat_candidates if col in df.columns), None)
    lon_col = next((col for col in lon_candidates if col in df.columns), None)
    if lat_col and lon_col:
        df = df.copy()
        df["lat"] = df[lat_col].astype(float)
        df["lon"] = df[lon_col].astype(float)
        return df.dropna(subset=["lat", "lon"])
    if "geometry" in df.columns:
        df = df.copy()
        geoms = df["geometry"].apply(lambda val: wkt.loads(val) if isinstance(val, str) and val else None)
        df["lat"] = geoms.apply(lambda geom: geom.y if geom is not None else None)
        df["lon"] = geoms.apply(lambda geom: geom.x if geom is not None else None)
        return df.dropna(subset=["lat", "lon"])
    raise ValueError("Unable to derive lat/lon columns for validation.")


def _combine_coordinates(frames: List[pd.DataFrame]) -> pd.DataFrame:
    coords = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        subset = frame[["lat", "lon"]].dropna()
        if not subset.empty:
            coords.append(subset)
    if not coords:
        return pd.DataFrame(columns=["lat", "lon"])
    return pd.concat(coords, ignore_index=True)


def _estimate_zoom(lat_span: float, lon_span: float) -> float:
    span = max(lat_span, lon_span, 1e-6)
    if span < 0.01:
        return 15
    if span < 0.05:
        return 13
    if span < 0.1:
        return 12
    if span < 0.5:
        return 10
    if span < 1.0:
        return 9
    if span < 2.0:
        return 8
    return 7


def _safe_json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            value = str(value)
    return value


def _df_to_feature_collection(df: pd.DataFrame) -> dict:
    features = []
    for _, row in df.iterrows():
        geom = None
        geom_wkt = row.get("geometry")
        if isinstance(geom_wkt, str) and geom_wkt:
            try:
                geom = wkt.loads(geom_wkt)
            except Exception:
                geom = None
        if geom is None:
            lat = row.get("lat") or row.get("start_node_lat")
            lon = row.get("lon") or row.get("start_node_lon")
            if pd.notna(lat) and pd.notna(lon):
                geom = Point(float(lon), float(lat))
        if geom is None or geom.is_empty:
            continue
        props = {}
        for col, value in row.items():
            if col == "geometry":
                continue
            props[col] = _safe_json_value(value)
        features.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return {"type": "FeatureCollection", "features": features}


def _load_trimmed_path_geojson(interstate: str, outputs_root: Path) -> Optional[dict]:
    path = (outputs_root / interstate / "clipped_mainline.geojson").resolve()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _path_coordinates_from_geojson(geojson: Optional[dict]) -> List[tuple[float, float]]:
    if not geojson:
        return []
    coords: List[tuple[float, float]] = []
    for feature in geojson.get("features", []):
        geom_payload = feature.get("geometry")
        if not geom_payload:
            continue
        try:
            geom = shape(geom_payload)
        except Exception:
            continue
        if geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            coords.extend(list(geom.coords))
        elif geom.geom_type == "MultiLineString":
            largest = None
            for candidate in geom.geoms:
                if largest is None or candidate.length > largest.length:
                    largest = candidate
            if largest is None:
                continue
            coords.extend(list(largest.coords))
    cleaned: List[tuple[float, float]] = []
    for coord in coords:
        if not cleaned or coord != cleaned[-1]:
            cleaned.append(coord)
    return cleaned


def _extract_endpoints(df: pd.DataFrame, trimmed_path_geojson: Optional[dict]) -> pd.DataFrame:
    path_coords = _path_coordinates_from_geojson(trimmed_path_geojson)
    if path_coords:
        start_lon, start_lat = path_coords[0]
        end_lon, end_lat = path_coords[-1]
        return pd.DataFrame(
            [
                {"endpoint": "Start", "lat": start_lat, "lon": start_lon},
                {"endpoint": "End", "lat": end_lat, "lon": end_lon},
            ]
        )
    if df.empty:
        return df
    ordering_start = "x_start_mile" if "x_start_mile" in df.columns else None
    ordering_end = "x_end_mile" if "x_end_mile" in df.columns else ordering_start
    start_df = df.sort_values(ordering_start) if ordering_start else df
    end_df = df.sort_values(ordering_end) if ordering_end else df
    start = start_df.iloc[0].copy()
    end = end_df.iloc[-1].copy()
    start["endpoint"] = "Start"
    end["endpoint"] = "End"
    return pd.DataFrame([start, end])


def _build_leaflet_html(center_lat: float, center_lon: float, zoom: float, layers: List[dict]) -> str:
    layers_payload = json.dumps(layers, default=lambda x: x)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>OSM Validation Viewer</title>
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
  <style>
    html, body {{
      height: 100%;
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    #map {{
      width: 100%;
      height: 92vh;
    }}
    .legend {{
      background: rgba(255, 255, 255, 0.95);
      padding: 12px;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.15);
      line-height: 1.4;
    }}
    .legend h4 {{
      margin: 0 0 8px;
      font-size: 14px;
      color: #0f172a;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.2);
    }}
  </style>
</head>
<body>
  <div id=\"map\"></div>
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <script>
    const map = L.map('map', {{ preferCanvas: true }}).setView([{center_lat:.6f}, {center_lon:.6f}], {zoom:.2f});
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const layersData = {layers_payload};

    function onEachFeature(feature, layer) {{
      if (!feature.properties) return;
      const rows = Object.entries(feature.properties)
        .filter(([_, value]) => value !== null && value !== undefined && value !== '')
        .map(([key, value]) => `<div><strong>${{key}}</strong>: ${{
          typeof value === 'object' ? JSON.stringify(value) : value
        }}</div>`);
      if (rows.length) {{
        layer.bindPopup(rows.join(''));
      }}
    }}

    function addLayer(def) {{
      if (!def.geojson || !def.geojson.features || def.geojson.features.length === 0) {{
        return null;
      }}
      return L.geoJSON(def.geojson, {{
        style: feature => {{
          if (!(feature.geometry && feature.geometry.type !== 'Point')) {{
            return undefined;
          }}
          const weight = typeof def.line_weight === 'number' ? def.line_weight : 3;
          return {{ color: def.color, weight, opacity: 0.9 }};
        }},
        pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{
          radius: 6,
          color: '#0f172a',
          weight: 1,
          fillColor: def.color,
          fillOpacity: 0.9
        }}),
        onEachFeature
      }}).addTo(map);
    }}

    const bounds = [];
    layersData.forEach(def => {{
      const layer = addLayer(def);
      if (layer && layer.getBounds && layer.getBounds().isValid()) {{
        bounds.push(layer.getBounds());
      }}
    }});

    if (bounds.length) {{
      const combined = bounds.slice(1).reduce((acc, b) => acc.extend(b), bounds[0]);
      if (combined.isValid()) {{
        map.fitBounds(combined.pad(0.12));
      }}
    }}

    const legend = L.control({{ position: 'bottomleft' }});
    legend.onAdd = function() {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML = `<h4>Layers</h4>` + layersData
        .map(def => `<div class="legend-item"><span class="legend-swatch" style="background:${{def.color}}"></span>${{def.name}}</div>`)
        .join('');
      return div;
    }};
    legend.addTo(map);
  </script>
</body>
</html>
"""


def generate_validation_kml(
    interstate: str,
    lanes_filename: str,
    ramps_filename: str,
    outputs_root: Path = OUTPUT_ROOT,
) -> Path:
    base_dir = outputs_root / interstate
    lanes_path = base_dir / lanes_filename
    ramps_path = base_dir / ramps_filename
    if not lanes_path.exists():
        raise FileNotFoundError(f"Lanes file not found: {lanes_filename}")
    if not ramps_path.exists():
        raise FileNotFoundError(f"Ramps file not found: {ramps_filename}")

    df_lanes = pd.read_csv(lanes_path)
    df_ramps = pd.read_csv(ramps_path)
    lanes_latlon = _prepare_lat_lon(df_lanes, ["start_node_lat", "lat"], ["start_node_lon", "lon"])
    endpoints_df = _extract_endpoints(
        lanes_latlon,
        _load_trimmed_path_geojson(interstate, outputs_root),
    )

    kml_dir = base_dir / "validation"
    kml_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    kml_path = kml_dir / f"validation_{timestamp}.kml"

    kml = Kml()

    for _, row in df_lanes.iterrows():
        coords = _row_to_coords(row)
        if not coords:
            continue
        if len(coords) == 1:
            pnt = kml.newpoint(coords=[coords[0]])
            pnt.style.iconstyle.color = Color.orange
            pnt.style.iconstyle.scale = 0.8
            _add_extended_data(pnt, row)
        else:
            line = kml.newlinestring(coords=coords)
            line.style.linestyle.width = 3
            line.style.linestyle.color = Color.orange
            _add_extended_data(line, row)

    for _, row in df_ramps.iterrows():
        coords = _row_to_coords(row)
        if not coords:
            continue
        color = Color.red
        entry = row.get("entry_node")
        exit_node = row.get("exit_node")
        entry_bool = str(entry).lower() == "true"
        exit_bool = str(exit_node).lower() == "true"
        if entry_bool and not exit_bool:
            color = Color.green
        elif entry_bool and exit_bool:
            color = Color.yellow
        pnt = kml.newpoint(coords=[coords[0]])
        pnt.style.iconstyle.color = color
        pnt.style.iconstyle.scale = 0.9
        _add_extended_data(pnt, row)

    for _, row in endpoints_df.iterrows():
        pnt = kml.newpoint(coords=[(row["lon"], row["lat"])])
        pnt.style.iconstyle.color = Color.yellow
        pnt.style.iconstyle.scale = 1.1
        for key, value in row.items():
            if key in {"lat", "lon"}:
                continue
            pnt.extendeddata.newdata(name=str(key), value=str(value))

    kml.save(kml_path)
    return kml_path


def _row_to_coords(row) -> List[tuple]:
    geom = None
    geom_wkt = row.get("geometry")
    if isinstance(geom_wkt, str) and geom_wkt:
        try:
            geom = wkt.loads(geom_wkt)
        except Exception:
            geom = None
    coords = []
    if geom is not None:
        if geom.geom_type == "Point":
            coords = [(geom.x, geom.y)]
        elif geom.geom_type in {"LineString", "LinearRing"}:
            coords = list(geom.coords)
        elif geom.geom_type == "MultiLineString" and len(geom.geoms):
            largest = max(geom.geoms, key=lambda g: g.length)
            coords = list(largest.coords)
    if not coords:
        lon = row.get("lon")
        lat = row.get("lat")
        if pd.notna(lon) and pd.notna(lat):
            coords = [(float(lon), float(lat))]
    return coords


def _add_extended_data(feature, row: pd.Series) -> None:
    for key, value in row.items():
        if key in {"geometry"}:
            continue
        if pd.isna(value):
            value = ""
        feature.extendeddata.newdata(name=str(key), value=str(value))


def generate_validation_osm_html(
    interstate: str,
    lanes_filename: str,
    ramps_filename: str,
    outputs_root: Path = OUTPUT_ROOT,
) -> Path:
    base_dir = outputs_root / interstate
    lanes_path = base_dir / lanes_filename
    ramps_path = base_dir / ramps_filename
    if not lanes_path.exists():
        raise FileNotFoundError(f"Lanes file not found: {lanes_filename}")
    if not ramps_path.exists():
        raise FileNotFoundError(f"Ramps file not found: {ramps_filename}")

    lanes_df = pd.read_csv(lanes_path)
    ramps_df = pd.read_csv(ramps_path)
    lanes_prepped = _prepare_lat_lon(lanes_df, ["start_node_lat", "lat"], ["start_node_lon", "lon"])
    ramps_prepped = _prepare_lat_lon(ramps_df, ["lat"], ["lon"])

    entry_mask = ramps_prepped["entry_node"].astype(str).str.lower() == "true"
    exit_mask = ramps_prepped["exit_node"].astype(str).str.lower() == "true"
    both_mask = entry_mask & exit_mask

    ramps_entry = pd.concat(
        [ramps_prepped[entry_mask & ~exit_mask], ramps_prepped[both_mask]],
        ignore_index=True,
    )
    ramps_exit = pd.concat(
        [ramps_prepped[exit_mask & ~entry_mask], ramps_prepped[both_mask]],
        ignore_index=True,
    )

    trimmed_path_geojson = _load_trimmed_path_geojson(interstate, outputs_root)
    endpoints_df = _extract_endpoints(lanes_prepped, trimmed_path_geojson)

    path_coords = _path_coordinates_from_geojson(trimmed_path_geojson)
    path_coords_df = (
        pd.DataFrame([{"lat": lat, "lon": lon} for lon, lat in path_coords])
        if path_coords
        else None
    )

    coord_sources = [lanes_prepped, ramps_entry, ramps_exit, endpoints_df, path_coords_df]
    coord_frame = _combine_coordinates(coord_sources)
    if coord_frame.empty:
        center_lat = 0.0
        center_lon = 0.0
        zoom = 8
    else:
        min_lat, max_lat = coord_frame["lat"].min(), coord_frame["lat"].max()
        min_lon, max_lon = coord_frame["lon"].min(), coord_frame["lon"].max()
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        zoom = _estimate_zoom(abs(max_lat - min_lat), abs(max_lon - min_lon))

    layers: List[dict] = []

    def _append_geojson_layer(geojson: Optional[dict], name: str, color: str, line_weight: Optional[float] = None) -> None:
        if not geojson:
            return
        features = geojson.get("features")
        if not features:
            return
        layer = {"name": name, "color": color, "geojson": geojson}
        if line_weight is not None:
            layer["line_weight"] = line_weight
        layers.append(layer)

    def _append_layer(df: pd.DataFrame, name: str, color: str, line_weight: Optional[float] = None) -> None:
        if df is None or df.empty:
            return
        geojson = _df_to_feature_collection(df)
        if not geojson["features"]:
            return
        layer = {"name": name, "color": color, "geojson": geojson}
        if line_weight is not None:
            layer["line_weight"] = line_weight
        layers.append(layer)

    _append_geojson_layer(trimmed_path_geojson, "Mainline Path", "#000000", line_weight=4)
    _append_layer(lanes_prepped, "Lane Add/Drop", "#f97316")
    _append_layer(ramps_entry, "On-Ramp", "#22c55e")
    _append_layer(ramps_exit, "Off-Ramp", "#dc2626")
    _append_layer(endpoints_df, "Segment Endpoint", "#facc15")

    if not layers:
        raise ValueError("No spatial features available; cannot build validation map.")

    html = _build_leaflet_html(center_lat, center_lon, zoom, layers)
    validation_dir = base_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = validation_dir / f"osm_validation_{timestamp}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


__all__ = [
    "generate_validation_kml",
    "generate_validation_osm_html",
]
