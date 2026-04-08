# metanet_mapping

This repository supports two connected workflows:

1. **Road-network extraction** for a corridor segment (lanes, ramps, validation artifacts).
2. **PeMS to METANET calibration/simulation** (time-space gridding, smoothing, parameter fitting, simulation diagnostics).

It includes a Python backend, a React frontend, CLI tools, and notebooks.

## Architecture Overview

- `segment_extractor/`:
  Core extraction pipeline (Overpass query, graph pathing, clipping, stationing, lane/ramp processing).
- `api/`:
  FastAPI app exposing extraction jobs and artifact endpoints.
- `frontend/`:
  React + Vite UI. Dev server proxies `/api/*` to `http://127.0.0.1:8000`.
- `get_road_network.py`:
  CLI entrypoint for network extraction.
- `pems/`:
  Matrix-building, smoothing, sparsity, and plotting helpers.
- `metanet_calibration/`:
  METANET dynamics and Pyomo/IPOPT calibration logic.
- `calibrate_pems.ipynb`:
  Main notebook for preprocessing, calibration, and simulation.
- `inputs/`, `outputs/`, `jobs/`:
  Local data/artifact directories.

## Prerequisites

### Python

- Python 3.10+
- Recommended: virtual environment (`venv` or conda)
- IPOPT solver for calibration (code currently points to `/opt/homebrew/bin/ipopt`)

Install core Python packages:

```bash
pip install numpy pandas scipy matplotlib seaborn torch ijson \
            geopandas shapely pyproj networkx overpy \
            fastapi uvicorn pydantic pyomo
```

### Frontend

- Node.js 18+
- npm

## Setup

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Then install Python dependencies (above).

### 2) Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Workflow A: Network Extraction

### Option 1: Via Frontend UI

Run in two terminals from repo root.

#### Terminal A: backend API

```bash
source .venv/bin/activate
uvicorn api.main:app --reload
```

Backend URL: `http://127.0.0.1:8000`

##### Terminal B: frontend

```bash
cd frontend
npm run dev
```

Frontend URL (Vite): usually `http://127.0.0.1:5173`

The frontend is configured to proxy `http://127.0.0.1:5173/api/*` to the backend.

### Option 2: via CLI

```bash
python get_road_network.py \
  --interstate "I-5" \
  --seg_start_lat 33.86 --seg_start_lon -118.00 \
  --seg_end_lat 33.78 --seg_end_lon -117.90 \
  --anchor_postmile 114.8 \
  --stationing_direction descending \
  --out_lanes_csv lanes.csv \
  --out_ramps_csv ramps.csv \
  --generate_validation both
```

Outputs land under `outputs/<interstate>/`; intermediates in `outputs/intermediates/`.

### Option 3: via API

Start backend, then use endpoints:

- `GET /healthz`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/files/{artifact}`
- `POST /snap`
- `POST /validate-network`
- `POST /validate-network/osm`

## Workflow B: PeMS Calibration + Simulation

Open and run:

- `calibrate_pems.ipynb`

Typical notebook flow:

1. Load station metadata and detector demand data.
2. Build speed/flow matrices in space-time.
3. Apply optional smoothing/imputation.
4. Construct `rho_hat`, `q_hat`, `v_hat`.
5. Run calibration (`metanet_calibration.ipopt_optimization.run_calibration`).
6. Run forward simulation (`metanet_calibration.metanet_dynamics.run_metanet_sim`).
7. Save arrays/figures under `outputs/`.

## Data and Git Hygiene

`.gitignore` excludes local runtime data/artifacts, including:

- `inputs/*`
- `outputs/*`
- `jobs/*`

Avoid committing large raw detector files (GitHub hard limit: 100 MB/file).

## Troubleshooting

### IPOPT not found

Calibration will fail if IPOPT is missing or at a different path. Update solver path in:

- `metanet_calibration/ipopt_optimization.py`

### Overpass timeout / rate limit

Extraction depends on Overpass API. If requests fail, retry with a smaller corridor/bbox.

### GMNS outputs

Each successful extraction now also writes a GMNS bundle under `outputs/<interstate>/gmns/`:

- `node.csv`
- `link.csv`
- `config.csv`
- `../gmns.zip`

The current pass encodes on/off ramp attachment points in `node.csv` via `node_type`, splits mainline links at those nodes, and omits optional GMNS columns when they have no data. Existing `lanes.csv` and `ramps.csv` outputs are still produced unchanged.

### Frontend cannot reach backend

- Ensure backend is running on `127.0.0.1:8000`.
- Ensure frontend is started from `frontend/` with `npm run dev`.
- Check `frontend/vite.config.js` proxy target.

## Suggested First Run

1. Start backend + frontend and submit one extraction job in the UI.
2. Run one CLI extraction to verify local pipeline output files.
3. Run `calibrate_pems.ipynb` on a short time window before full runs.
