"""Схемы галереи."""
from pydantic import BaseModel


class GalleryItem(BaseModel):
    gallery_id: str
    camera_group: int
    thumb_url: str | None = None
    frame_url: str | None = None
    attention_url: str | None = None


class GalleryPage(BaseModel):
    total: int
    offset: int
    items: list[GalleryItem]


class Neighbor(BaseModel):
    gallery_id: str
    confidence: float
    camera_group: int
    same_scene: bool


class GallerySimilar(BaseModel):
    gallery_id: str
    camera_group: int
    neighbors: list[Neighbor]
