"""Схемы эмбеддинга."""
from pydantic import BaseModel


class EmbedResponse(BaseModel):
    dim: int
    embedding: list[float]
