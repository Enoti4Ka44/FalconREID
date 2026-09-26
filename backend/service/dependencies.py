"""Dependency Injection: все зависимости для роутеров."""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Header, HTTPException

from .config import Settings, get_settings

if TYPE_CHECKING:
    from .engine import ReIDEngine
    from .gallery_index import GalleryIndex
    from .reid_inference import ReIDInference

# ── глобальные синглтоны (инициализируются в lifespan) ──────────────

_engine: ReIDEngine | None = None
_gallery_index: GalleryIndex | None = None
_reid_inference: ReIDInference | None = None
_usage_service = None
_watchlist_service = None
_search_service = None
_gallery_service = None
_detector_service = None


def init_services(settings: Settings) -> None:
    """Вызывается один раз при старте приложения (в lifespan)."""
    global _engine, _gallery_index, _reid_inference
    global _usage_service, _watchlist_service, _search_service
    global _gallery_service, _detector_service

    from .engine import ReIDEngine
    from .gallery_index import GalleryIndex
    from .reid_inference import ReIDInference
    from .services import (
        GalleryService,
        SearchService,
        UsageService,
        WatchlistService,
        DetectorService,
    )

    # Загружаем компоненты отдельно (для новых зависимостей)
    _gallery_index = GalleryIndex(settings.gallery_index, settings.data_root, settings.threshold)
    _reid_inference = ReIDInference(settings.model_path, settings.gallery_index)

    # Facade для обратной совместимости
    _engine = ReIDEngine(
        settings.model_path,
        settings.gallery_index,
        settings.data_root,
        settings.threshold,
    )

    _usage_service = UsageService()
    _watchlist_service = WatchlistService(_gallery_index)
    _search_service = SearchService(_engine, _watchlist_service, _usage_service)
    _gallery_service = GalleryService(_engine)
    _detector_service = DetectorService()


def shutdown_services() -> None:
    """Вызывается при остановке приложения."""
    pass  # watchlist сохраняется при каждой мутации


# ── Depends-функции ──────────────────────────────────────────────────


def get_engine() -> ReIDEngine:
    assert _engine is not None, "Engine not initialized"
    return _engine


def get_gallery_index():
    from .gallery_index import GalleryIndex
    assert _gallery_index is not None, "GalleryIndex not initialized"
    return _gallery_index


def get_reid_inference():
    from .reid_inference import ReIDInference
    assert _reid_inference is not None, "ReIDInference not initialized"
    return _reid_inference


def get_search_service() -> SearchService:
    assert _search_service is not None, "SearchService not initialized"
    return _search_service


def get_gallery_service() -> GalleryService:
    assert _gallery_service is not None, "GalleryService not initialized"
    return _gallery_service


def get_watchlist_service() -> WatchlistService:
    assert _watchlist_service is not None, "WatchlistService not initialized"
    return _watchlist_service


def get_usage_service() -> UsageService:
    assert _usage_service is not None, "UsageService not initialized"
    return _usage_service


def get_detector_service() -> DetectorService:
    assert _detector_service is not None, "DetectorService not initialized"
    return _detector_service


# ── фундамент для auth ──────────────────────────────────────────────


async def verify_api_key(
    settings: Settings = Depends(get_settings),
    x_api_key: str | None = Header(None),
) -> None:
    """Заготовка для API-ключей. Пока ничего не проверяет если ключ не задан."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
