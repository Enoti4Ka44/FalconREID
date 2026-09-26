"""Схемы статистики."""
from pydantic import BaseModel


class LocationStat(BaseModel):
    camera_group: int
    count: int


class TrackStats(BaseModel):
    threshold: float
    n_tracks: int
    tracks: list[dict]


class UsageRecord(BaseModel):
    t: float
    ms: float
    refused: bool
    accepted: int
    n_images: int
    alerts: int


class UsageStats(BaseModel):
    requests: int
    avg_ms: float | None
    p95_ms: float | None
    refusal_rate: float | None
    alerts_total: int
    recent: list[dict]


class TopCamera(BaseModel):
    camera_group: int
    count: int


class UserStats(BaseModel):
    requests: int
    avg_ms: float | None
    p95_ms: float | None
    refusal_rate: float | None
    alerts_total: int
    top_cameras: list[TopCamera]


class Top3Candidate(BaseModel):
    gallery_id: str
    confidence: float
    camera_group: int


class HistoryItem(BaseModel):
    id: str
    refused: bool
    n_accepted: int
    inference_ms: float
    n_images: int
    alerts: int
    top3: list[Top3Candidate]
    created_at: float
