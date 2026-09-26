"""Сервис галереи: страницы, похожие."""
from PIL import Image

from ..engine import ReIDEngine
from ..exceptions import NotFoundError


class GalleryService:
    def __init__(self, engine: ReIDEngine) -> None:
        self._engine = engine

    def page(self, group: int | None = None, offset: int = 0,
             limit: int = 60) -> dict:
        return self._engine.gallery_page(group, offset, min(limit, 200))

    def similar(self, image_id: str, top_k: int = 12) -> dict:
        res = self._engine.similar_in_gallery(image_id, top_k=top_k)
        if res is None:
            raise NotFoundError("unknown image_id")
        return res

    def thumb_path(self, image_id: str) -> str:
        p = self._engine.thumb_path(image_id)
        if p is None:
            raise NotFoundError("unknown image_id")
        return p

    def frame_path(self, image_id: str, data_root) -> str:
        p = data_root / "images" / f"{image_id}.jpg"
        if not p.exists():
            raise NotFoundError("unknown image_id")
        return p

    def attention_for_image(self, img: Image.Image) -> bytes:
        """Attention map для произвольного изображения (используется в /api/explain)."""
        return self._engine.attention_map(img)
