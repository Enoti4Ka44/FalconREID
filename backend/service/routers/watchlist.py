"""Роутер: вочлист + попарные сравнения."""
import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import get_current_user
from ..db import get_db
from ..dependencies import get_engine, get_watchlist_service
from ..exceptions import BadRequestError, NotFoundError
from ..models_db import User
from ..schemas.watchlist import WatchAdd

router = APIRouter(prefix="/api", tags=["watchlist"])


@router.get(
    "/watchlist",
    summary="Список watchlist",
    description="Возвращает все объекты текущего пользователя в watchlist "
                "(«на контроле»). При каждом поиске автоматически проверяется "
                "сходство запроса с объектами watchlist.\n\n"
                "**Используется для:**\n"
                "- Мониторинга разыскиваемых ТС\n"
                "- Получения алертов при совпадении\n\n"
                "**Ответ:**\n"
                "- `items` — список объектов с id, name, gallery_id, created\n\n"
                "**Требует аутентификации.**",
)
async def watchlist_get(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_watchlist_service),
) -> dict:
    items = await svc.list_items(db, user.id)
    return {"items": items}


@router.post(
    "/watchlist",
    summary="Добавить в watchlist",
    description="Добавляет объект галереи в watchlist текущего пользователя. "
                "При каждом поиске будет автоматически проверяться сходство "
                "запроса с этим объектом.\n\n"
                "**Параметры:**\n"
                "- `gallery_id` — идентификатор объекта из галереи\n"
                "- `name` — произвольное имя (опционально)\n\n"
                "**Ответ:**\n"
                "- `id` — идентификатор записи watchlist\n"
                "- `gallery_id` — подтверждение добавленного объекта\n\n"
                "**Требует аутентификации.**",
    responses={404: {"description": "Объект не найден в галерее"}},
)
async def watchlist_add(
    item: WatchAdd,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_watchlist_service),
) -> dict:
    return await svc.add(db, user.id, item.gallery_id, item.name)


@router.delete(
    "/watchlist/{wid}",
    summary="Удалить из watchlist",
    description="Удаляет объект из watchlist текущего пользователя.\n\n"
                "**Параметры:**\n"
                "- `wid` — идентификатор записи watchlist\n\n"
                "**Требует аутентификации.**",
    responses={404: {"description": "Запись не найдена"}},
)
async def watchlist_del(
    wid: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_watchlist_service),
) -> dict:
    return await svc.remove(db, user.id, wid)


@router.post(
    "/pair_matrix",
    summary="Матрица попарных сходств",
    description="Возвращает матрицу попарных косинусных сходств для списка "
                "объектов галереи. Используется для визуализации сходства "
                "между кандидатами.\n\n"
                "**Параметры:**\n"
                "- `ids` — список gallery_id (от 2 до 30)\n\n"
                "**Ответ:**\n"
                "- `ids` — исходный список\n"
                "- `matrix` — матрица NxN сходств\n\n"
                "**Требует аутентификации.**",
    responses={
        400: {"description": "Количество ids вне диапазона 2..30"},
        404: {"description": "Один из gallery_id не найден"},
    },
)
def pair_matrix(
    ids: list[str],
    engine=Depends(get_engine),
) -> dict:
    if not (2 <= len(ids) <= 30):
        raise BadRequestError("2..30 ids")
    m = engine.pair_matrix(ids)
    if m is None:
        raise NotFoundError("unknown gallery_id in list")
    return {"ids": ids, "matrix": m}


@router.get(
    "/compare",
    summary="Сравнение двух объектов",
    description="Возвращает косинусное сходство между двумя объектами галереи.\n\n"
                "**Параметры:**\n"
                "- `a` — gallery_id первого объекта\n"
                "- `b` — gallery_id второго объекта\n\n"
                "**Ответ:**\n"
                "- `a, b` — исходные идентификаторы\n"
                "- `confidence` — косинусное сходство (0..1)\n\n"
                "**Требует аутентификации.**",
    responses={404: {"description": "Один из gallery_id не найден"}},
)
def compare(
    a: str,
    b: str,
    engine=Depends(get_engine),
) -> dict:
    va, vb = engine.vector_of(a), engine.vector_of(b)
    if va is None or vb is None:
        raise NotFoundError("unknown gallery_id")
    return {"a": a, "b": b, "confidence": round(float(np.dot(va, vb)), 4)}
