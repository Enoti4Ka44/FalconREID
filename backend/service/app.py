"""FALCON ReID — сервис формирования цифрового признака ТС.

FastAPI-микросервис: принимает изображение + BBox, возвращает эмбеддинг,
топ-N кандидатов из галереи с оценками уверенности либо отказ.
OpenAPI доступен на /docs (Swagger UI) и /openapi.json.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import close_db, init_db, seed_db
from .dependencies import init_services, shutdown_services
from .exceptions import register_exception_handlers
from .routers import (
    auth_router,
    detect_router,
    gallery_router,
    health_router,
    search_router,
    stats_router,
    watchlist_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_db()
    await seed_db(settings)
    init_services(settings)
    yield
    shutdown_services()
    await close_db()


app = FastAPI(
    title="FALCON ReID API",
    description="Сервис формирования цифрового признака транспортного средства "
                "для сопоставления снимков одного автомобиля из разных локаций "
                "без использования государственного номера.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

for router in [
    auth_router,
    detect_router,
    gallery_router,
    health_router,
    search_router,
    stats_router,
    watchlist_router,
]:
    app.include_router(router)
