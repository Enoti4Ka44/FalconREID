"""Вочлист: контроль объектов галереи с персистентностью в PostgreSQL."""
import uuid

import numpy as np
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..exceptions import NotFoundError
from ..gallery_index import GalleryIndex
from ..models_db import WatchlistItem


class WatchlistService:
    def __init__(self, gallery_index: GalleryIndex) -> None:
        self._gallery = gallery_index

    async def list_items(self, db: AsyncSession, user_id: uuid.UUID) -> list[dict]:
        result = await db.execute(
            select(WatchlistItem)
            .where(WatchlistItem.user_id == user_id)
            .order_by(WatchlistItem.created_at.desc())
        )
        items = result.scalars().all()
        return [
            {
                "id": str(item.id),
                "name": item.name,
                "gallery_id": item.gallery_id,
                "created": item.created_at.strftime("%Y-%m-%d %H:%M"),
            }
            for item in items
        ]

    async def add(
        self, db: AsyncSession, user_id: uuid.UUID, gallery_id: str, name: str = ""
    ) -> dict:
        v = self._gallery.vector_of(gallery_id)
        if v is None:
            raise NotFoundError("unknown gallery_id")

        item = WatchlistItem(
            user_id=user_id,
            gallery_id=gallery_id,
            name=name or gallery_id[:8],
            embedding=v.astype(np.float32).tobytes(),
        )
        db.add(item)
        await db.commit()
        await db.refresh(item)
        return {"id": str(item.id), "gallery_id": gallery_id}

    async def remove(self, db: AsyncSession, user_id: uuid.UUID, wid: str) -> dict:
        try:
            wid_uuid = uuid.UUID(wid)
        except ValueError:
            raise NotFoundError("unknown watch id")
        result = await db.execute(
            select(WatchlistItem).where(
                WatchlistItem.id == wid_uuid,
                WatchlistItem.user_id == user_id,
            )
        )
        item = result.scalar_one_or_none()
        if item is None:
            raise NotFoundError("unknown watch id")
        await db.delete(item)
        await db.commit()
        return {"id": wid}

    async def check_alerts(
        self, db: AsyncSession, user_id: uuid.UUID, embedding: np.ndarray, threshold: float
    ) -> list[dict]:
        result = await db.execute(
            select(WatchlistItem).where(WatchlistItem.user_id == user_id)
        )
        items = result.scalars().all()
        alerts = []
        for item in items:
            # вектор цели берётся из ТЕКУЩЕЙ галереи: после смены моделей
            # сохранённый в БД вектор устаревает, а размерность может совпасть,
            # и сравнение молча шло бы со старым признаком (дефект Д18)
            w_emb = self._gallery.vector_of(item.gallery_id)
            if w_emb is None:
                w_emb = np.frombuffer(item.embedding, dtype=np.float32)
            if w_emb.shape != embedding.shape:
                continue
            s = float(np.dot(embedding, w_emb))
            if s >= threshold:
                alerts.append({
                    "id": str(item.id),
                    "name": item.name,
                    "gallery_id": item.gallery_id,
                    "confidence": round(s, 4),
                })
        alerts.sort(key=lambda a: -a["confidence"])
        return alerts[:5]
