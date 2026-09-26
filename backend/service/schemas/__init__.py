"""Pydantic-модели для API."""
from .auth import (
    APIKeyCreate,
    APIKeyInfo,
    APIKeyResponse,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)
from .common import ErrorResponse, HealthResponse
from .detect import BBox, DetectResponse
from .embed import EmbedResponse
from .gallery import GalleryItem, GalleryPage, GallerySimilar, Neighbor
from .search import SearchRequest, SearchResponse, SearchResult
from .stats import LocationStat, TrackStats, UsageStats
from .watchlist import WatchAdd, WatchlistItem

__all__ = [
    "APIKeyCreate",
    "APIKeyInfo",
    "APIKeyResponse",
    "TokenResponse",
    "UserLogin",
    "UserRegister",
    "UserResponse",
    "ErrorResponse",
    "HealthResponse",
    "BBox",
    "DetectResponse",
    "EmbedResponse",
    "GalleryItem",
    "GalleryPage",
    "GallerySimilar",
    "Neighbor",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "LocationStat",
    "TrackStats",
    "UsageStats",
    "WatchAdd",
    "WatchlistItem",
]
