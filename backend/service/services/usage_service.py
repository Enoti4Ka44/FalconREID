"""Аудит-лог: персистентность в PostgreSQL."""
import statistics as st
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models_db import SearchHistory


class UsageService:
    async def record(
        self,
        db: AsyncSession,
        *,
        ms: float,
        refused: bool,
        accepted: int,
        n_images: int,
        alerts: int,
        user_id: uuid.UUID | None = None,
        top3: list[dict] | None = None,
    ) -> None:
        entry = SearchHistory(
            user_id=user_id,
            image_ids=[],
            refused=refused,
            n_accepted=accepted,
            inference_ms=ms,
            n_images=n_images,
            alerts=alerts,
            top3=top3 or [],
        )
        db.add(entry)
        await db.commit()

    async def stats(self, db: AsyncSession) -> dict:
        result = await db.execute(select(SearchHistory).order_by(SearchHistory.created_at.desc()))
        records = result.scalars().all()

        if not records:
            return {
                "requests": 0,
                "avg_ms": None,
                "p95_ms": None,
                "refusal_rate": None,
                "alerts_total": 0,
                "recent": [],
            }

        ms_values = [r.inference_ms for r in records]
        p95 = sorted(ms_values)[int(len(ms_values) * 0.95)] if len(ms_values) >= 5 else None

        return {
            "requests": len(records),
            "avg_ms": round(st.mean(ms_values), 1),
            "p95_ms": round(p95, 1) if p95 is not None else None,
            "refusal_rate": round(sum(1 for r in records if r.refused) / len(records), 3),
            "alerts_total": sum(r.alerts for r in records),
            "recent": [
                {
                    "t": r.created_at.timestamp(),
                    "ms": r.inference_ms,
                    "refused": r.refused,
                    "accepted": r.n_accepted,
                    "n_images": r.n_images,
                    "alerts": r.alerts,
                }
                for r in records[:40]
            ],
        }

    async def user_stats(self, db: AsyncSession, user_id: uuid.UUID) -> dict:
        result = await db.execute(
            select(SearchHistory)
            .where(SearchHistory.user_id == user_id)
            .order_by(SearchHistory.created_at.desc())
        )
        records = result.scalars().all()

        if not records:
            return {
                "requests": 0,
                "avg_ms": None,
                "p95_ms": None,
                "refusal_rate": None,
                "alerts_total": 0,
                "top_cameras": [],
            }

        ms_values = [r.inference_ms for r in records]
        p95 = sorted(ms_values)[int(len(ms_values) * 0.95)] if len(ms_values) >= 5 else None

        camera_counts: dict[int, int] = {}
        for r in records:
            for c in r.top3:
                cg = c.get("camera_group")
                if cg is not None:
                    camera_counts[cg] = camera_counts.get(cg, 0) + 1
        top_cameras = sorted(
            [{"camera_group": k, "count": v} for k, v in camera_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:10]

        return {
            "requests": len(records),
            "avg_ms": round(st.mean(ms_values), 1),
            "p95_ms": round(p95, 1) if p95 is not None else None,
            "refusal_rate": round(sum(1 for r in records if r.refused) / len(records), 3),
            "alerts_total": sum(r.alerts for r in records),
            "top_cameras": top_cameras,
        }

    async def user_history(
        self, db: AsyncSession, user_id: uuid.UUID, limit: int = 20
    ) -> list[dict]:
        result = await db.execute(
            select(SearchHistory)
            .where(SearchHistory.user_id == user_id)
            .order_by(SearchHistory.created_at.desc())
            .limit(limit)
        )
        records = result.scalars().all()
        return [
            {
                "id": str(r.id),
                "refused": r.refused,
                "n_accepted": r.n_accepted,
                "inference_ms": r.inference_ms,
                "n_images": r.n_images,
                "alerts": r.alerts,
                "top3": r.top3,
                "created_at": r.created_at.timestamp(),
            }
            for r in records
        ]
