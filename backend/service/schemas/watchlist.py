"""Схемы вочлиста."""
from pydantic import BaseModel


class WatchAdd(BaseModel):
    gallery_id: str
    name: str = ""


class WatchlistItem(BaseModel):
    id: str
    name: str
    gallery_id: str
    created: str
