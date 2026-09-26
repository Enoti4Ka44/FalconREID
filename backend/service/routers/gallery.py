"""Роутер: галерея (список, миниатюры, кадры, attention, похожие)."""
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, Response

from ..config import Settings
from ..dependencies import get_gallery_service, get_settings
from ..exceptions import NotFoundError
from ..schemas.gallery import GalleryPage, GallerySimilar

router = APIRouter(prefix="/api/gallery", tags=["gallery"])


@router.get(
    "/list",
    response_model=GalleryPage,
    summary="Список галереи с пагинацией",
    description="Возвращает пагинированный список объектов галереи. "
                "Каждый элемент содержит gallery_id, camera_group и URL-ы "
                "для получения миниатюры, полного кадра и attention-карты.\n\n"
                "**Параметры:**\n"
                "- `group` — фильтр по камере-группе (опционально)\n"
                "- `offset` — смещение (по умолчанию 0)\n"
                "- `limit` — количество элементов (по умолчанию 60, макс 200)\n\n"
                "**Ответ:**\n"
                "- `total` — общее количество элементов\n"
                "- `offset` — текущее смещение\n"
                "- `items` — список объектов с URL-ами к ресурсам",
)
def gallery_list(
    group: int | None = Query(None, description="Фильтр по камере-группе"),
    offset: int = Query(0, description="Смещение пагинации"),
    limit: int = Query(60, description="Количество элементов (макс 200)"),
    svc=Depends(get_gallery_service),
) -> GalleryPage:
    page = svc.page(group=group, offset=offset, limit=limit)
    for item in page["items"]:
        gid = item["gallery_id"]
        item["thumb_url"] = f"/api/gallery/{gid}/thumb"
        item["frame_url"] = f"/api/gallery/{gid}/frame"
        item["attention_url"] = f"/api/gallery/{gid}/attention"
    return page


@router.get(
    "/{image_id}/thumb",
    summary="Миниатюра кандидата",
    description="Возвращает JPEG миниатюру (кроп) транспортного средства из галереи. "
                "Используется для отображения в карточках кандидатов.\n\n"
                "**Параметры:**\n"
                "- `image_id` — идентификатор изображения из галереи\n\n"
                "**Ответ:** JPEG изображение",
    responses={404: {"description": "Миниатюра не найдена"}},
)
def gallery_thumb(
    image_id: str,
    svc=Depends(get_gallery_service),
) -> FileResponse:
    p = svc.thumb_path(image_id)
    return FileResponse(p, media_type="image/jpeg")


@router.get(
    "/{image_id}/frame",
    summary="Полный кадр",
    description="Возвращает полный кадр (исходное изображение с камеры) для указанного "
                "объекта галереи. Используется для детального просмотра.\n\n"
                "**Параметры:**\n"
                "- `image_id` — идентификатор изображения из галереи\n\n"
                "**Ответ:** JPEG изображение полного кадра",
    responses={404: {"description": "Кадр не найден"}},
)
def gallery_frame(
    image_id: str,
    settings: Settings = Depends(get_settings),
    svc=Depends(get_gallery_service),
) -> FileResponse:
    p = svc.frame_path(image_id, settings.data_root)
    return FileResponse(p, media_type="image/jpeg")


@router.get(
    "/{image_id}/attention",
    summary="Attention-карта кандидата",
    description="Возвращает precomputed attention-карту (heatmap) для объекта галереи "
                "в формате PNG. Карта показывает, на какие области изображения "
                "обращает внимание нейросеть.\n\n"
                "**Важно:** Для работы необходимо заранее сгенерировать карты:\n"
                "```\npython -m src.build_gallery --attn\n```\n\n"
                "**Параметры:**\n"
                "- `image_id` — идентификатор изображения из галереи\n\n"
                "**Ответ:** PNG изображение attention-карты",
    responses={404: {"description": "Attention-карта не сгенерирована"}},
)
def gallery_attention(
    image_id: str,
    settings: Settings = Depends(get_settings),
) -> Response:
    p = settings.data_root / "attn" / f"{image_id}.png"
    if not p.exists():
        raise NotFoundError("attention map not found (run build_gallery --attn)")
    return Response(content=p.read_bytes(), media_type="image/png")


@router.get(
    "/{image_id}/similar",
    response_model=GallerySimilar,
    summary="Досье: похожие объекты",
    description="Возвращает список ближайших соседей объекта в галерее "
                "по косинусному сходству эмбеддингов. Используется для:\n"
                "- Просмотра «досье» ТС — где ещё появлялась эта машина\n"
                "- Нахождения «близнецов» — похожих, но разных ТС\n"
                "- Проверки качества идентификации\n\n"
                "**Параметры:**\n"
                "- `image_id` — идентификатор изображения из галереи\n"
                "- `top_k` — количество соседей (по умолчанию 12)\n\n"
                "**Ответ:**\n"
                "- `gallery_id` — исходный объект\n"
                "- `camera_group` — камера-группа исходного объекта\n"
                "- `neighbors` — список соседей с confidence и same_scene",
    responses={404: {"description": "Объект не найден в галерее"}},
)
def gallery_similar(
    image_id: str,
    top_k: int = Query(12, description="Количество соседей"),
    svc=Depends(get_gallery_service),
) -> GallerySimilar:
    return svc.similar(image_id, top_k=top_k)
