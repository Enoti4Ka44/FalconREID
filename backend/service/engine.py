"""Движок инференса — facade над GalleryIndex + ReIDInference.

Для обратной совместимости: существующий код использует ReIDEngine,
который внутри делегирует в соответствующие компоненты.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .gallery_index import GalleryIndex
from .reid_inference import ReIDInference


class ReIDEngine:
    """Ансамбль моделей + галерея эмбеддингов.

    Facade: делегирует в GalleryIndex (без torch) и ReIDInference (с torch).
    """

    def __init__(self, model_path: str, gallery_index: Path, data_root: Path,
                 threshold: float, model2_path: str | None = None, w2: float = 0.5):
        self._gallery = GalleryIndex(gallery_index, data_root, threshold)
        self._inference = ReIDInference(model_path, gallery_index, model2_path, w2)

    # --- Свойства (проксируем) ---

    @property
    def device(self) -> str:
        return self._inference.device

    @property
    def threshold(self) -> float:
        return self._gallery.threshold

    @threshold.setter
    def threshold(self, value: float):
        self._gallery.threshold = value

    @property
    def gallery_size(self) -> int:
        return self._gallery.gallery_size

    @property
    def n_models(self) -> int:
        return self._inference.n_models

    @property
    def gallery_mat(self) -> np.ndarray:
        return self._gallery.gallery_mat

    @property
    def gallery_ids(self):
        return self._gallery.gallery_ids

    @property
    def gallery_groups(self):
        return self._gallery.gallery_groups

    @property
    def model(self):
        return self._inference.model

    @property
    def tf(self):
        return self._inference.tf

    # --- Методы галереи (без torch) ---

    def search(self, emb: np.ndarray, top_k: int = 10, threshold=None):
        return self._gallery.search(emb, top_k, threshold)

    def similar_in_gallery(self, image_id: str, top_k: int = 12):
        return self._gallery.similar_in_gallery(image_id, top_k)

    def location_stats(self):
        return self._gallery.location_stats()

    def gallery_page(self, group: int | None = None, offset: int = 0, limit: int = 60):
        return self._gallery.gallery_page(group, offset, limit)

    def cross_location_tracks(self, thr: float = 0.65, min_locs: int = 2):
        return self._gallery.cross_location_tracks(thr, min_locs)

    def pair_matrix(self, ids: list[str]):
        return self._gallery.pair_matrix(ids)

    def vector_of(self, gallery_id: str):
        return self._gallery.vector_of(gallery_id)

    def thumb_path(self, image_id: str):
        return self._gallery.thumb_path(image_id)

    def gallery_crop(self, image_id: str):
        return self._gallery.gallery_crop(image_id)

    # --- Методы инференса (с torch) ---

    @staticmethod
    def crop_bbox(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
        return ReIDInference.crop_bbox(img, x, y, w, h)

    def embed(self, img: Image.Image) -> np.ndarray:
        return self._inference.embed(img)

    def attention_map(self, img: Image.Image) -> bytes:
        return self._inference.attention_map(img)
