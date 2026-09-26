"""Роутер: статистика (локации, треки, аудит)."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import get_current_user
from ..db import get_db
from ..dependencies import get_engine, get_usage_service
from ..models_db import User
from ..schemas.stats import HistoryItem, UsageStats, UserStats

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get(
    "/locations",
    summary="Статистика локаций",
    description="Возвращает количество снимков на каждую камеру-группу. "
                "Камеры-группы восстановлены по фону кадра и геометрии BBox "
                "без какой-либо разметки.\n\n"
                "**Используется для:**\n"
                "- Визуализации покрытия камер на карте\n"
                "- Фильтрации галереи по локациям\n"
                "- Анализа распределения данных\n\n"
                "**Ответ:**\n"
                "- `locations` — список {camera_group, count}\n"
                "- `total` — общее количество объектов в галерее\n\n"
                "**Не требует аутентификации.**",
)
def stats_locations(engine=Depends(get_engine)) -> dict:
    return {"locations": engine.location_stats(), "total": engine.gallery_size}


@router.get(
    "/tracks",
    summary="Треки ТС по локациям",
    description="Возвращает «треки» — связные компоненты галереи по косинусному "
                "сходству >= порога, замеченные в 2+ разных локациях. "
                "Это машины, которые повторно появлялись в разных частях города.\n\n"
                "**Используется для:**\n"
                "- Поиска «перемещающихся» ТС\n"
                "- Расследования маршрутов\n"
                "- Визуализации на карте\n\n"
                "**Ответ:**\n"
                "- `threshold` — порог сходства\n"
                "- `n_tracks` — количество треков\n"
                "- `tracks` — список треков с участниками и локациями\n\n"
                "**Не требует аутентификации.**",
)
def stats_tracks(engine=Depends(get_engine)) -> dict:
    return engine.cross_location_tracks()


@router.get(
    "/usage",
    response_model=UsageStats,
    summary="Общая статистика использования",
    description="Возвращает агрегированную статистику по всем поисковым запросам: "
                "количество запросов, среднее время инференса, p95, "
                "доля отказов, количество алертов watchlist.\n\n"
                "**Ответ:**\n"
                "- `requests` — общее количество запросов\n"
                "- `avg_ms, p95_ms` — время инференса\n"
                "- `refusal_rate` — доля отказов\n"
                "- `alerts_total` — количество алертов\n"
                "- `recent` — последние 40 записей\n\n"
                "**Требует аутентификации.**",
)
async def stats_usage(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_usage_service),
) -> UsageStats:
    return await svc.stats(db)


@router.get(
    "/my",
    response_model=UserStats,
    summary="Моя статистика",
    description="Возвращает статистику поисковых запросов текущего пользователя: "
                "количество запросов, среднее время инференса, p95, "
                "доля отказов, количество алертов, топ камеры-группы.\n\n"
                "**Ответ:**\n"
                "- `requests` — количество запросов пользователя\n"
                "- `avg_ms, p95_ms` — время инференса\n"
                "- `refusal_rate` — доля отказов\n"
                "- `alerts_total` — количество алертов\n"
                "- `top_cameras` — топ камеры-групп по частоте появления в результатах\n\n"
                "**Требует аутентификации.**",
)
async def stats_my(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_usage_service),
) -> UserStats:
    return await svc.user_stats(db, user.id)


@router.get(
    "/my/history",
    response_model=list[HistoryItem],
    summary="Моя история поисков",
    description="Возвращает последние поисковые запросы текущего пользователя "
                "с топ-3 кандидатами в каждом результате.\n\n"
                "**Параметры:**\n"
                "- `limit` — количество записей (по умолчанию 20, макс 100)\n\n"
                "**Ответ (каждый элемент):**\n"
                "- `id` — идентификатор запроса\n"
                "- `refused` — был ли отказ\n"
                "- `n_accepted` — количество принятых кандидатов\n"
                "- `inference_ms` — время инференса\n"
                "- `top3` — топ-3 кандидата (gallery_id, confidence, camera_group)\n"
                "- `created_at` — время запроса (Unix timestamp)\n\n"
                "**Требует аутентификации.**",
)
async def stats_my_history(
    limit: int = Query(20, description="Количество записей (макс 100)"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_usage_service),
) -> list[HistoryItem]:
    return await svc.user_history(db, user.id, limit=min(limit, 100))
