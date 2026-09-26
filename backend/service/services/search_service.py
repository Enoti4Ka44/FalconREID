"""Сервис поиска ТС: embed → search → watchlist check → audit."""
import io
import time
import uuid

import numpy as np
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from ..engine import ReIDEngine
from ..exceptions import BadRequestError
from ..schemas.search import SearchResponse, SearchResult
from .usage_service import UsageService
from .watchlist_service import WatchlistService


class SearchService:
    def __init__(
        self,
        engine: ReIDEngine,
        watchlist: WatchlistService,
        usage: UsageService,
    ) -> None:
        self._engine = engine
        self._watchlist = watchlist
        self._usage = usage

    @staticmethod
    def decode_image(raw: bytes) -> Image.Image:
        if not raw:
            raise BadRequestError("empty file")
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise BadRequestError("cannot decode image")

    def maybe_crop(
        self, img: Image.Image,
        x: int | None, y: int | None,
        w: int | None, h: int | None,
    ) -> Image.Image:
        if any(v is not None for v in (x, y, w, h)):
            if None in (x, y, w, h) or w <= 0 or h <= 0:
                raise BadRequestError("bbox requires all of x,y,w,h with w,h > 0")
            if x >= img.width or y >= img.height:
                raise BadRequestError("bbox outside image bounds")
            return self._engine.crop_bbox(img, x, y, w, h)
        return img

    def embed(self, img: Image.Image) -> np.ndarray:
        return self._engine.embed(img)

    async def search(
        self,
        images: list[bytes],
        x: int | None = None,
        y: int | None = None,
        w: int | None = None,
        h: int | None = None,
        top_k: int = 10,
        threshold: float | None = None,
        db: AsyncSession | None = None,
        user_id: uuid.UUID | None = None,
    ) -> SearchResponse:
        if len(images) > 5:
            raise BadRequestError("не более 5 снимков в мульти-запросе")

        t0 = time.perf_counter()
        embs = []
        for i, raw in enumerate(images):
            img = self.decode_image(raw)
            if i == 0:
                img = self.maybe_crop(img, x, y, w, h)
            embs.append(self._engine.embed(img))

        emb = np.mean(embs, axis=0)
        emb /= np.linalg.norm(emb)
        res = self._engine.search(emb, top_k=top_k, threshold=threshold)
        dt = (time.perf_counter() - t0) * 1000

        thr_now = threshold if threshold is not None else self._engine.threshold

        # watchlist check (async, требует db + user_id)
        alerts = []
        if db is not None and user_id is not None:
            alerts = await self._watchlist.check_alerts(db, user_id, emb, thr_now)

        if db is not None:
            top3 = [
                {"gallery_id": c["gallery_id"], "confidence": c["confidence"], "camera_group": c["camera_group"]}
                for c in res["candidates"][:3]
            ]
            await self._usage.record(
                db, ms=dt, refused=res["refused"], accepted=res["n_accepted"],
                n_images=len(images), alerts=len(alerts), user_id=user_id,
                top3=top3,
            )

        return SearchResponse(
            watchlist_alerts=alerts,
            query_embedding_dim=len(emb),
            inference_ms=round(dt, 1),
            threshold=threshold if threshold is not None else self._engine.threshold,
            refused=res["refused"],
            message=(
                "Уверенного совпадения в галерее не найдено — отказ от идентификации"
                if res["refused"]
                else f"Принято кандидатов: {res['n_accepted']}"
            ),
            n_query_images=len(images),
            sim_max=res["sim_max"],
            sim_hist=res["sim_hist"],
            candidates=[SearchResult(**c) for c in res["candidates"]],
        )

    async def batch_search(
        self,
        images: list[bytes],
        top_k: int = 10,
        threshold: float | None = None,
        db: AsyncSession | None = None,
        user_id: uuid.UUID | None = None,
    ) -> list[SearchResponse]:
        """Пакетный поиск: каждый файл — отдельный запрос."""
        if len(images) > 32:
            raise BadRequestError("не более 32 файлов в batch")

        results = []
        for raw in images:
            result = await self.search(
                [raw],
                top_k=top_k,
                threshold=threshold,
                db=db,
                user_id=user_id,
            )
            results.append(result)
        return results
