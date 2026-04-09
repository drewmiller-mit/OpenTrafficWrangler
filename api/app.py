"""
FastAPI application exposing the extraction pipeline.
"""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import overpy

from segment_extractor.geometry import snap_point_to_lines
from segment_extractor.graph import ways_by_ids_to_gdf
from segment_extractor.io import overpass_query_snapping_bbox
from segment_extractor.pipeline import run_pipeline
from segment_extractor.validation_utils import (
    generate_validation_kml,
    generate_validation_osm_html,
)

from .schemas import (
    ArtifactInfo,
    JobCreateResponse,
    JobStatus,
    JobSummary,
    PathMode,
    RunRequest,
    SnapRequest,
    SnapResponse,
    ValidationRequest,
    ValidationResponse,
)
from .storage import JobStore

OUTPUT_ROOT = Path("outputs")
INTERMEDIATE_ROOT = OUTPUT_ROOT / "intermediates"
JOB_STORE_ROOT = Path("jobs")


def create_app() -> FastAPI:
    app = FastAPI(title="Metanet Segment Extractor", version="0.1.0")
    store = JobStore(JOB_STORE_ROOT)
    OSM_FEATURE_CACHE: Dict[tuple, tuple[float, dict]] = {}
    OSM_FEATURE_CACHE_TTL = 60.0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_ROOT.mkdir(parents=True, exist_ok=True)

    def _artifact_from_path(path: Path) -> ArtifactInfo:
        stat = path.stat()
        return ArtifactInfo(
            name=path.name,
            relative_path=str(path.relative_to(OUTPUT_ROOT)).replace("\\", "/"),
            size_bytes=stat.st_size,
            updated_at=datetime.fromtimestamp(stat.st_mtime),
            mime_type=mimetypes.guess_type(path.name)[0],
        )

    def _list_intermediates() -> List[ArtifactInfo]:
        INTERMEDIATE_ROOT.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for child in INTERMEDIATE_ROOT.rglob("*"):
            if child.is_file():
                artifacts.append(_artifact_from_path(child))
        artifacts.sort(key=lambda item: item.updated_at, reverse=True)
        return artifacts

    def _clear_intermediates() -> int:
        if not INTERMEDIATE_ROOT.exists():
            return 0
        deleted = 0
        for child in INTERMEDIATE_ROOT.iterdir():
            try:
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
                deleted += 1
            except Exception:
                continue
        return deleted

    def _resolve_output_path(relative_path: str) -> Path:
        base = OUTPUT_ROOT.resolve()
        target = (OUTPUT_ROOT / relative_path).resolve()
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        if base not in target.parents and target != base:
            raise HTTPException(status_code=400, detail="Invalid artifact path.")
        return target

    def _fetch_osm_features(
        south: float,
        west: float,
        north: float,
        east: float,
        *,
        ref_regex: str | None = None,
    ) -> overpy.Result:
        if south >= north or west >= east:
            raise HTTPException(status_code=400, detail="Invalid bounding box.")
        if (north - south) > 1.0 or (east - west) > 1.0:
            raise HTTPException(status_code=400, detail="Bounding box too large; zoom in further.")
        api = overpy.Overpass()
        filters = '["highway"]'
        if ref_regex:
            filters += f'["ref"~"{ref_regex}"]'
        query = f"""
        [out:json][timeout:90];
        (
          way{filters}({south},{west},{north},{east});
        );
        (._;>;);
        out body;
        """
        try:
            return api.query(query)
        except overpy.exception.OverpassTooManyRequests as exc:
            raise HTTPException(
                status_code=429,
                detail="Overpass server is experiencing high load. Please wait a moment and zoom in further if possible.",
            ) from exc
        except overpy.exception.OverpassGatewayTimeout as exc:
            raise HTTPException(
                status_code=504,
                detail="Overpass timed out while fetching OSM features. Try zooming in and retrying.",
            ) from exc

    def _osm_features_to_geojson(result: overpy.Result) -> dict:
        features = []
        for way in result.ways:
            coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
            if len(coords) < 2:
                continue
            properties = {"id": int(way.id)}
            properties.update(way.tags)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": properties,
                }
            )
        return {"type": "FeatureCollection", "features": features}


    def _generate_validation_kml(interstate: str, lanes_filename: str, ramps_filename: str) -> Path:
        try:
            path = generate_validation_kml(
                interstate,
                lanes_filename,
                ramps_filename,
                outputs_root=OUTPUT_ROOT,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return path.relative_to(OUTPUT_ROOT)

    def _generate_validation_osm_html(interstate: str, lanes_filename: str, ramps_filename: str) -> Path:
        try:
            path = generate_validation_osm_html(
                interstate,
                lanes_filename,
                ramps_filename,
                outputs_root=OUTPUT_ROOT,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return path.relative_to(OUTPUT_ROOT)

    def _record(job_id: str) -> JobSummary:
        data = store.get(job_id)
        return JobSummary(
            job_id=job_id,
            status=data.status,
            message=data.message,
            outputs=data.outputs,
            logs=data.logs,
        )

    def _log(job_id: str, message: str) -> None:
        store.append_log(job_id, message)

    def _run_job(job_id: str, payload: RunRequest) -> None:
        store.update(job_id, status=JobStatus.running)
        _log(job_id, "Job started.")
        job_intermediate_dir = INTERMEDIATE_ROOT / job_id
        if job_intermediate_dir.exists():
            shutil.rmtree(job_intermediate_dir, ignore_errors=True)
        job_intermediate_dir.mkdir(parents=True, exist_ok=True)
        try:
            _log(job_id, "Calling pipeline.")
            run_pipeline(
                interstate_name=payload.interstate,
                seg_start_lat=payload.seg_start_lat,
                seg_start_lon=payload.seg_start_lon,
                seg_end_lat=payload.seg_end_lat,
                seg_end_lon=payload.seg_end_lon,
                out_lanes_csv=payload.out_lanes_csv,
                out_ramps_csv=payload.out_ramps_csv,
                anchor_postmile=payload.anchor_postmile,
                end_postmile=payload.end_postmile,
                bbox_buffer_ft=payload.bbox_buffer_ft,
                path_mode=payload.path_mode.value,
                ref_list=payload.ref_list,
                start_osm_node=payload.start_node,
                end_osm_node=payload.end_node,
                intermediate_dir=job_intermediate_dir,
                stationing_direction=payload.stationing_direction.value,
                allow_relaxation=payload.allow_relaxation,
                log_fn=lambda message: _log(job_id, message),
            )
            interstate_dir = OUTPUT_ROOT / payload.interstate
            outputs = {
                "lanes_csv": str(interstate_dir / payload.out_lanes_csv),
                "ramps_csv": str(interstate_dir / payload.out_ramps_csv),
                "gmns_archive": str(interstate_dir / "gmns.zip"),
                "intermediate_dir": str(job_intermediate_dir),
            }
            store.update(
                job_id,
                status=JobStatus.finished,
                outputs=outputs,
                message="Completed successfully.",
            )
            _log(job_id, "Pipeline completed successfully.")
        except Exception as exc:  # pragma: no cover - bubble up to API
            detail = str(exc) or repr(exc)
            store.update(
                job_id,
                status=JobStatus.failed,
                message=detail,
                outputs={"intermediate_dir": str(job_intermediate_dir)},
            )
            _log(job_id, f"Job failed: {detail}")
            _log(job_id, f"Intermediate artifacts saved under {job_intermediate_dir}")

    @app.get("/healthz")
    def healthcheck() -> dict:
        return {"status": "ok"}

    @app.post("/jobs", response_model=JobCreateResponse, status_code=202)
    def create_job(request: RunRequest, background_tasks: BackgroundTasks) -> JobCreateResponse:
        job_id = uuid.uuid4().hex
        store.create(job_id, request)
        _log(job_id, "Job queued.")
        background_tasks.add_task(_run_job, job_id, request)
        return JobCreateResponse(job_id=job_id, status=JobStatus.queued)

    @app.get("/jobs/{job_id}", response_model=JobSummary)
    def get_job(job_id: str) -> JobSummary:
        try:
            return _record(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found.")

    @app.post("/snap", response_model=SnapResponse)
    def snap_coordinate(request: SnapRequest) -> SnapResponse:
        try:
            result = overpass_query_snapping_bbox(
                request.lat,
                request.lon,
                request.lat,
                request.lon,
                request.bbox_buffer_deg,
            )
            way_ids = [int(w.id) for w in result.ways if w.tags.get("highway")]
            if not way_ids:
                raise ValueError("No highway-tagged geometry found near that point.")
            gdf = ways_by_ids_to_gdf(result, way_ids)
            if gdf.empty:
                raise ValueError("No highway-tagged geometry found near that point.")
            snapped_lat, snapped_lon = snap_point_to_lines(gdf, request.lat, request.lon)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc))

        return SnapResponse(lat=snapped_lat, lon=snapped_lon)

    @app.get("/jobs/{job_id}/files/{artifact}")
    def download_output(job_id: str, artifact: str, inline: bool = Query(False)):
        try:
            record = store.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found.")
        if not record.outputs:
            raise HTTPException(status_code=404, detail="Outputs not available yet.")
        path_str = record.outputs.get(artifact)
        if not path_str:
            raise HTTPException(status_code=404, detail="Artifact not found for this job.")
        path = Path(path_str)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact file missing on disk.")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        response = FileResponse(path, media_type=media_type)
        disposition = "inline" if inline else "attachment"
        response.headers["Content-Disposition"] = f'{disposition}; filename="{path.name}"'
        return response

    @app.get("/intermediates", response_model=List[ArtifactInfo])
    def list_intermediate_outputs() -> List[ArtifactInfo]:
        return _list_intermediates()

    @app.delete("/intermediates")
    def clear_intermediate_outputs() -> dict:
        deleted = _clear_intermediates()
        return {"deleted": deleted}

    @app.get("/artifacts/file")
    def download_artifact(
        path: str = Query(..., description="Path relative to outputs/"),
        inline: bool = Query(False, description="If true, return with inline disposition"),
    ):
        file_path = _resolve_output_path(path)
        media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        response = FileResponse(file_path, media_type=media_type)
        disposition = "inline" if inline else "attachment"
        response.headers["Content-Disposition"] = f'{disposition}; filename="{file_path.name}"'
        return response

    @app.get("/osm/features")
    def get_osm_features(
        south: float = Query(..., description="South latitude of bounding box"),
        west: float = Query(..., description="West longitude of bounding box"),
        north: float = Query(..., description="North latitude of bounding box"),
        east: float = Query(..., description="East longitude of bounding box"),
        interstate: str | None = Query(None, description="Optional route filter, e.g., 'I-24' or 'CA SR 55'"),
    ):
        try:
            ref_regex = normalize_interstate_name(interstate) if interstate else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        cache_key = (
            round(south, 4),
            round(west, 4),
            round(north, 4),
            round(east, 4),
            ref_regex or "",
        )
        now = time.time()
        cached = OSM_FEATURE_CACHE.get(cache_key)
        if cached and (now - cached[0]) < OSM_FEATURE_CACHE_TTL:
            return cached[1]
        try:
            result = _fetch_osm_features(south, west, north, east, ref_regex=ref_regex)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        geojson = _osm_features_to_geojson(result)
        OSM_FEATURE_CACHE[cache_key] = (now, geojson)
        if len(OSM_FEATURE_CACHE) > 200:
            oldest_key = min(OSM_FEATURE_CACHE.items(), key=lambda item: item[1][0])[0]
            OSM_FEATURE_CACHE.pop(oldest_key, None)
        return geojson

    @app.post("/validate-network", response_model=ValidationResponse)
    def validate_network(request: ValidationRequest) -> ValidationResponse:
        try:
            artifact_rel = _generate_validation_kml(
                request.interstate, request.lanes_filename, request.ramps_filename
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return ValidationResponse(status="ready", artifact_path=str(artifact_rel).replace("\\", "/"))

    @app.post("/validate-network/osm", response_model=ValidationResponse)
    def validate_network_osm(request: ValidationRequest) -> ValidationResponse:
        try:
            artifact_rel = _generate_validation_osm_html(
                request.interstate, request.lanes_filename, request.ramps_filename
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return ValidationResponse(status="ready", artifact_path=str(artifact_rel).replace("\\", "/"))

    return app
