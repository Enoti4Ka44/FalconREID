"""Роутер: здоровье и кривые валидации."""
import json

from fastapi import APIRouter, Depends

from ..config import Settings
from ..dependencies import get_engine, get_settings
from ..exceptions import NotFoundError
from ..schemas.common import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка здоровья сервиса",
    description="Возвращает текущий статус сервиса: состояние, размер галереи, "
                "активный порог отказа, количество моделей в ансамбле.\n\n"
                "**Используется для:**\n"
                "- Мониторинга доступности сервиса\n"
                "- Проверки конфигурации перед запросами\n"
                "- Healthcheck в Docker/Kubernetes\n\n"
                "**Не требует аутентификации.**",
)
def health(
    engine=Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=engine.device,
        gallery_size=engine.gallery_size,
        threshold=engine.threshold,
        models=engine.n_models,
    )


@router.get(
    "/val_curves",
    summary="Кривые валидации",
    description="Возвращает кривые F1, TNR, Precision, Recall по сетке порогов "
                "для визуализации в интерфейсе.\n\n"
                "**Используется для:**\n"
                "- Умного слайдера порога в UI\n"
                "- Выбора оптимального порога отказа\n"
                "- Обоснования выбора порога на защите\n\n"
                "**Ответ:**\n"
                "- `grid` — сетка значений порога\n"
                "- `F1, TNR, precision, recall` — значения метрик\n\n"
                "**Не требует аутентификации.**",
    responses={404: {"description": "Кривые валидации не найдены в модели"}},
)
def val_curves(settings: Settings = Depends(get_settings)) -> dict:
    path = settings.val_curves_path
    if not path.exists():
        raise NotFoundError("validation curves not bundled")
    d = json.loads(path.read_text(encoding="utf-8"))
    return {
        "grid": d["grid"],
        "F1": d["F1"],
        "TNR": d["TNR"],
        "precision": d["precision"],
        "recall": d["recall"],
    }
