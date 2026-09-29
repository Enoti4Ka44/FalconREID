"""Сервисный слой."""
from .detect_service import DetectorService
from .usage_service import UsageService

__all__ = [
    "DetectorService",
    "GalleryService",
    "SearchService",
    "UsageService",
    "WatchlistService",
]


def __getattr__(name: str):
    if name == "GalleryService":
        from .gallery_service import GalleryService
        return GalleryService
    if name == "SearchService":
        from .search_service import SearchService
        return SearchService
    if name == "WatchlistService":
        from .watchlist_service import WatchlistService
        return WatchlistService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
