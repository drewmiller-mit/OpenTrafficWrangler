"""
Disk-backed job storage utilities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import JobStatus, RunRequest


@dataclass
class JobRecord:
    job_id: str
    payload: RunRequest
    status: JobStatus = JobStatus.queued
    message: Optional[str] = None
    outputs: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "payload": self.payload.dict(),
            "status": self.status.value,
            "message": self.message,
            "outputs": self.outputs,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "logs": self.logs,
        }

    @classmethod
    def from_path(cls, path: Path) -> "JobRecord":
        data = json.loads(path.read_text())
        payload = RunRequest(**data["payload"])
        record = cls(
            job_id=data["job_id"],
            payload=payload,
            status=JobStatus(data["status"]),
            message=data.get("message"),
            outputs=data.get("outputs"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            logs=data.get("logs", []),
        )
        return record


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, JobRecord] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for path in self.root.glob("*.json"):
            record = JobRecord.from_path(path)
            self._cache[record.job_id] = record

    def _path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def create(self, job_id: str, payload: RunRequest) -> JobRecord:
        record = JobRecord(job_id=job_id, payload=payload)
        self._cache[job_id] = record
        self._persist(record)
        return record

    def update(self, job_id: str, **changes) -> JobRecord:
        record = self._cache[job_id]
        for key, value in changes.items():
            setattr(record, key, value)
        record.updated_at = datetime.utcnow()
        self._persist(record)
        return record

    def get(self, job_id: str) -> JobRecord:
        return self._cache[job_id]

    def list_ids(self):
        return list(self._cache.keys())

    def _persist(self, record: JobRecord) -> None:
        path = self._path_for(record.job_id)
        path.write_text(json.dumps(record.to_dict(), indent=2))

    def append_log(self, job_id: str, message: str) -> JobRecord:
        record = self._cache[job_id]
        timestamp = datetime.utcnow().isoformat()
        record.logs.append(f"{timestamp} - {message}")
        record.updated_at = datetime.utcnow()
        self._persist(record)
        return record
