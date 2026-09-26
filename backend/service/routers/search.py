"""Роутер: поиск ТС, эмбеддинг, интерпретируемость."""
import base64

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import get_current_user
from ..db import get_db
from ..dependencies import get_gallery_service, get_search_service
from ..models_db import User
from ..schemas.embed import EmbedResponse
from ..schemas.search import SearchResponse

router = APIRouter(prefix="/api", tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Поиск транспортного средства",
    description="Принимает 1–5 изображений одного ТС (с BBox) и возвращает топ-N кандидатов "
                "из галереи с оценками уверенности, либо отказ если уверенного совпадения нет.\n\n"
                "**Параметры:**\n"
                "- `files` — 1–5 JPEG/PNG изображений одного ТС\n"
                "- `x, y, w, h` — координаты BBox (применяется к первому изображению)\n"
                "- `top_k` — количество кандидатов (по умолчанию 10)\n"
                "- `threshold` — переопределить порог отказа\n\n"
                "**Ответ содержит:**\n"
                "- `candidates` — список кандидатов с gallery_id, confidence, accepted\n"
                "- `refused` — true если ни один кандидат не прошёл порог\n"
                "- `inference_ms` — время инференса в миллисекундах\n"
                "- `sim_hist` — гистограмма близостей (для UI)\n"
                "- `watchlist_alerts` — совпадения с watchlist",
)
async def search(
    files: list[UploadFile] = File(..., description="1–5 снимков одного ТС"),
    x: int | None = Form(None, description="X-координата левого верхнего угла BBox"),
    y: int | None = Form(None, description="Y-координата левого верхнего угла BBox"),
    w: int | None = Form(None, description="Ширина BBox в пикселях"),
    h: int | None = Form(None, description="Высота BBox в пикселях"),
    top_k: int = Form(10, description="Количество кандидатов для возврата"),
    threshold: float | None = Form(None, description="Переопределить порог отказа"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_search_service),
) -> SearchResponse:
    raw_files = [await f.read() for f in files]
    return await svc.search(
        raw_files, x=x, y=y, w=w, h=h, top_k=top_k, threshold=threshold,
        db=db, user_id=user.id,
    )


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Извлечение эмбеддинга",
    description="Принимает изображение ТС (с опциональным BBox) и возвращает "
                "3840-мерный float32 вектор признака. Используется для кастомной "
                "постобработки или интеграций.\n\n"
                "**Параметры:**\n"
                "- `file` — JPEG/PNG изображение\n"
                "- `x, y, w, h` — координаты BBox (опционально)\n\n"
                "**Ответ:**\n"
                "- `dim` — размерность вектора (3840)\n"
                "- `embedding` — массив float значений",
)
async def embed_only(
    file: UploadFile = File(..., description="Изображение ТС"),
    x: int | None = Form(None, description="X-координата BBox"),
    y: int | None = Form(None, description="Y-координата BBox"),
    w: int | None = Form(None, description="Ширина BBox"),
    h: int | None = Form(None, description="Высота BBox"),
    user: User = Depends(get_current_user),
    svc=Depends(get_search_service),
) -> EmbedResponse:
    raw = await file.read()
    img = svc.decode_image(raw)
    img = svc.maybe_crop(img, x, y, w, h)
    emb = svc.embed(img)
    return EmbedResponse(dim=len(emb), embedding=[round(float(v), 6) for v in emb])


@router.post(
    "/explain",
    summary="Карта внимания модели",
    description="Принимает изображение ТС и возвращает attention-карту (heatmap) "
                "в формате base64 PNG. Показывает, на какие области изображения "
                "обращает внимание нейросеть при извлечении признаков.\n\n"
                "**Используется для:**\n"
                "- Интерпретируемости решения\n"
                "- Проверки, что модель не опирается на номерные знаки\n"
                "- Отладки и анализа\n\n"
                "**Параметры:**\n"
                "- `file` — JPEG/PNG изображение\n"
                "- `x, y, w, h` — координаты BBox (опционально)\n\n"
                "**Ответ:**\n"
                "- `image_base64` — PNG изображение attention-карты в base64",
)
async def explain(
    file: UploadFile = File(..., description="Изображение ТС"),
    x: int | None = Form(None, description="X-координата BBox"),
    y: int | None = Form(None, description="Y-координата BBox"),
    w: int | None = Form(None, description="Ширина BBox"),
    h: int | None = Form(None, description="Высота BBox"),
    user: User = Depends(get_current_user),
    svc=Depends(get_search_service),
    gallery_svc=Depends(get_gallery_service),
) -> dict:
    raw = await file.read()
    img = svc.decode_image(raw)
    img = svc.maybe_crop(img, x, y, w, h)
    png = gallery_svc.attention_for_image(img)
    return {"image_base64": base64.b64encode(png).decode()}


@router.post(
    "/batch_search",
    response_model=list[SearchResponse],
    summary="Пакетный поиск ТС",
    description="Принимает массив изображений (до 32) и возвращает "
                "результат поиска для каждого изображения отдельно.\n\n"
                "**Используется для:**\n"
                "- Пакетной обработки видеопотока\n"
                "- FPS-тестирования производительности\n"
                "- Массовой идентификации ТС\n\n"
                "**Параметры:**\n"
                "- `files` — до 32 JPEG/PNG изображений ТС\n"
                "- `top_k` — количество кандидатов (по умолчанию 10)\n"
                "- `threshold` — переопределить порог отказа\n\n"
                "**Ответ:** массив SearchResponse (по одному на файл)\n\n"
                "**Пример ответа:**\n"
                "```json\n"
                "[\n"
                "  {\n"
                "    \"query_embedding_dim\": 3840,\n"
                "    \"inference_ms\": 42.3,\n"
                "    \"threshold\": 0.302,\n"
                "    \"refused\": false,\n"
                "    \"message\": \"Принято кандидатов: 5\",\n"
                "    \"n_query_images\": 1,\n"
                "    \"sim_max\": 0.85,\n"
                "    \"sim_hist\": {\"0.0-0.1\": 0, \"0.1-0.2\": 2},\n"
                "    \"candidates\": [\n"
                "      {\"gallery_id\": \"abc\", \"confidence\": 0.85, \"accepted\": true, \"camera_group\": 1}\n"
                "    ],\n"
                "    \"watchlist_alerts\": []\n"
                "  }\n"
                "]\n"
                "```\n\n"
                "**Требует аутентификации.**",
)
async def batch_search(
    files: list[UploadFile] = File(..., description="До 32 изображений ТС"),
    top_k: int = Form(10, description="Количество кандидатов"),
    threshold: float | None = Form(None, description="Порог отказа"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc=Depends(get_search_service),
) -> list[SearchResponse]:
    raw_files = [await f.read() for f in files]
    return await svc.batch_search(
        raw_files, top_k=top_k, threshold=threshold,
        db=db, user_id=user.id,
    )
