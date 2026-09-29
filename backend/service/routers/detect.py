"""Роутер: детекция ТС (YOLOv8)."""
from fastapi import APIRouter, Depends, File, UploadFile

from ..auth import get_current_user
from ..config import Settings
from ..dependencies import get_detector_service, get_search_service, get_settings
from ..models_db import User
from ..schemas.detect import DetectResponse

router = APIRouter(prefix="/api", tags=["detect"])


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Детекция транспортных средств",
    description="Принимает изображение и возвращает bounding box'ы обнаруженных "
                "транспортных средств с использованием YOLOv8n.\n\n"
                "**Используется для:**\n"
                "- UX-подсветки ТС на кадре в интерфейсе\n"
                "- Автоматического определения BBox для поиска\n\n"
                "**Важно:** По ТЗ детекция НЕ входит в задачу оценки. "
                "Официальные артефакты считаются по выданным BBox, "
                "детектор — только удобство интерфейса.\n\n"
                "**Параметры:**\n"
                "- `file` — JPEG/PNG изображение (полный кадр с камеры)\n\n"
                "**Ответ:**\n"
                "- `width, height` — размеры изображения\n"
                "- `boxes` — список обнаруженных ТС с координатами, score и классом",
)
async def detect_vehicles(
    file: UploadFile = File(..., description="Изображение с камеры"),
    settings: Settings = Depends(get_settings),
    user: User = Depends(get_current_user),
    svc=Depends(get_search_service),
    detector=Depends(get_detector_service),
) -> DetectResponse:
    raw = await file.read()
    img = svc.decode_image(raw)
    result = detector.detect(img, str(settings.gallery_index.parent / "yolov8n.pt"))
    return DetectResponse(**result)
