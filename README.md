# osm2macrosim

This repository is a freeway network extraction and detector-stationing workspace. The current codebase is centered on:

- extracting a freeway corridor from OpenStreetMap,
- exporting lane, ramp, and GMNS artifacts,
- serving that workflow through a FastAPI backend and React frontend,
- and supporting detector stationing work in the notebook [Get_Detector_Stationing.ipynb](./Get_Detector_Stationing.ipynb).

It does not currently contain the broader PeMS calibration / METANET package structure described in older versions of this README.

## Repo Layout

- `segment_extractor/`
  Core extraction logic: Overpass queries, graph construction, path isolation, clipping, stationing, ramp processing, visualization, and GMNS export.
- `api/`
  FastAPI server for running extraction jobs, listing artifacts, generating validation files, and inspecting nearby OSM features.
- `frontend/`
  React + Vite UI for selecting a corridor on a map and launching extraction jobs against the API.
- `get_road_network.py`
  Command-line entrypoint for running the extraction pipeline without the UI.
- `Get_Detector_Stationing.ipynb`
  Notebook for detector stationing work.
- `jobs/`
  Disk-backed API job records.
- `outputs/`
  Generated extraction artifacts.

## What Gets Produced

Both the CLI and API write extracted network artifacts under:

`outputs/<interstate>/`

Typical outputs include:

- `lanes.csv`
- `ramps.csv`
- `gmns.zip`

Intermediate debug artifacts are written under:

`outputs/intermediates/`

For API jobs, intermediates are grouped by job id under that directory.

## Prerequisites

- Python `3.10+`
- Node.js `18+`
- `npm`

Python packages used by this repo include:

- `fastapi`
- `uvicorn`
- `pydantic`
- `numpy`
- `pandas`
- `scipy`
- `geopandas`
- `shapely`
- `pyproj`
- `networkx`
- `overpy`
- `matplotlib`
- `contextily`
- `simplekml`
- `jupyter` if you want to run the notebook

## Setup

### 1. Create a Python environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install Python dependencies

```bash
pip install fastapi uvicorn pydantic numpy pandas scipy geopandas shapely pyproj \
  networkx overpy matplotlib contextily simplekml jupyter
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Startup

If you want the full app experience, run the backend and frontend in separate terminals.

### Terminal 1: start the API

From the repo root:

```bash
source .venv/bin/activate
uvicorn api.main:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

Health check:

`http://127.0.0.1:8000/healthz`

### Terminal 2: start the frontend

```bash
cd frontend
npm run dev
```

The frontend will usually be available at:

`http://127.0.0.1:5173`

The Vite dev server proxies `/api/*` requests to `http://127.0.0.1:8000`, so both processes need to be running for the UI to work.

## First Run

1. Start the API.
2. Start the frontend.
3. Open `http://127.0.0.1:5173`.
4. Pick start and end points on the map.
5. Submit an extraction job.
6. Download the generated `lanes.csv`, `ramps.csv`, or `gmns.zip` artifacts from the job panel.

## Command-Line Usage

You can run the network extractor directly without the frontend:

```bash
python get_road_network.py \
  --interstate "I-5" \
  --seg_start_lat 33.86 \
  --seg_start_lon -118.00 \
  --seg_end_lat 33.78 \
  --seg_end_lon -117.90 \
  --anchor_postmile 114.8 \
  --stationing_direction descending \
  --out_lanes_csv lanes.csv \
  --out_ramps_csv ramps.csv \
  --generate_validation both
```

Useful options:

- `--start_node` and `--end_node` to override inferred OSM nodes
- `--path_mode normal|prefer|avoid`
- `--ref_list ...` to influence path selection
- `--end_postmile` for two-point station calibration
- `--generate_validation none|kml|osm|both`

## API Endpoints

Main endpoints exposed by the backend:

- `GET /healthz`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/files/{artifact}`
- `POST /snap`
- `GET /intermediates`
- `DELETE /intermediates`
- `GET /artifacts/file`
- `GET /osm/features`
- `POST /validate-network`
- `POST /validate-network/osm`

## Detector Stationing Notebook

To work with the notebook:

```bash
source .venv/bin/activate
jupyter notebook
```

Then open:

`Get_Detector_Stationing.ipynb`

## Troubleshooting

### Frontend cannot reach the backend

- Confirm the API is running on `127.0.0.1:8000`.
- Confirm the frontend is running from `frontend/` with `npm run dev`.
- Confirm [frontend/vite.config.js](./frontend/vite.config.js) still proxies `/api` to `127.0.0.1:8000`.

### Overpass requests fail or time out

- Retry the request.
- Use a shorter corridor.
- Zoom further in before using OSM inspection in the UI.

### Files are not where you expect

- CLI and API outputs go directly under `outputs/`.
- Job metadata is written to `jobs/`.
