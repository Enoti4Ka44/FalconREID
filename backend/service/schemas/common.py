"""Общие схемы."""
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    device: str
    gallery_size: int
    threshold: float
    models: int
