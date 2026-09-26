"""Схемы поиска."""
from pydantic import BaseModel


class SearchResult(BaseModel):
    gallery_id: str
    confidence: float
    accepted: bool
    camera_group: int


class SearchResponse(BaseModel):
    query_embedding_dim: int
    inference_ms: float
    threshold: float
    refused: bool
    message: str
    n_query_images: int
    sim_max: float
    sim_hist: dict
    candidates: list[SearchResult]
    watchlist_alerts: list[dict] = []


class SearchRequest(BaseModel):
    top_k: int = 10
    threshold: float | None = None
