"""Роутеры API."""

from .auth import router as auth_router
from .detect import router as detect_router
from .gallery import router as gallery_router
from .health import router as health_router
from .search import router as search_router
from .stats import router as stats_router
from .watchlist import router as watchlist_router


__all__ = [
    "auth_router",
    "detect_router",
    "gallery_router",
    "health_router",
    "search_router",
    "stats_router",
    "watchlist_router",
]