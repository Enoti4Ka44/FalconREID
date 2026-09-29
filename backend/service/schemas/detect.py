"""Схемы детекции."""
from pydantic import BaseModel


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    score: float
    cls: str


class DetectResponse(BaseModel):
    width: int
    height: int
    boxes: list[BBox]
