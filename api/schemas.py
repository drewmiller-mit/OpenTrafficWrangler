"""
Pydantic models shared by the API endpoints.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class PathMode(str, Enum):
    normal = "normal"
    prefer = "prefer"
    avoid = "avoid"


class StationingDirection(str, Enum):
    ascending = "ascending"
    descending = "descending"


class RunRequest(BaseModel):
    interstate: str = Field(..., description="Route name, e.g., 'I-24' or 'CA SR 55'.")
    seg_start_lat: float
    seg_start_lon: float
    seg_end_lat: float
    seg_end_lon: float
    start_node: Optional[int] = Field(
        None,
        description="Optional OSM node ID override for the start coordinate.",
    )
    end_node: Optional[int] = Field(
        None,
        description="Optional OSM node ID override for the end coordinate.",
    )
    out_lanes_csv: str = Field("lanes.csv", description="Filename for exported lanes CSV.")
    out_ramps_csv: str = Field("ramps.csv", description="Filename for exported ramps CSV.")
    anchor_postmile: float = Field(0.0, description="Anchor postmile applied during stationing.")
    end_postmile: Optional[float] = Field(
        None,
        description="Optional ending postmile for two-point linear station calibration.",
    )
    bbox_buffer_ft: float = Field(
        0.08,
        description="Buffer applied to the Overpass bounding box (degrees, roughly).",
    )
    path_mode: PathMode = PathMode.normal
    stationing_direction: StationingDirection = StationingDirection.ascending
    ref_list: Optional[List[str]] = Field(
        None,
        description="List of references for prefer/avoid modes.",
    )
    allow_relaxation: bool = Field(
        False,
        description="Deprecated; the pipeline now relaxes automatically when strict extraction fails.",
    )

    @validator("interstate")
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("interstate must be non-empty.")
        return cleaned


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    finished = "finished"
    failed = "failed"


class JobSummary(BaseModel):
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    outputs: Optional[dict] = None
    logs: Optional[List[str]] = None


class JobCreateResponse(JobSummary):
    pass


class SnapRequest(BaseModel):
    interstate: str = Field(..., description="Route name to snap against.")
    lat: float
    lon: float
    bbox_buffer_deg: float = Field(
        0.05,
        description="Bounding-box buffer (degrees) used when fetching nearby geometry.",
    )


class SnapResponse(BaseModel):
    lat: float
    lon: float


class ArtifactInfo(BaseModel):
    name: str
    relative_path: str = Field(..., description="Path relative to outputs/")
    size_bytes: int
    updated_at: datetime
    mime_type: Optional[str] = None


class ValidationRequest(BaseModel):
    interstate: str
    lanes_filename: str
    ramps_filename: str


class ValidationResponse(BaseModel):
    status: str
    artifact_path: str
