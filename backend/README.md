# ФАЛЬКОН·ReID — цифровой признак транспортного средства

Решение кейса «Фалькон Тех» (ЛЦТ-2026): сервис формирования цифрового признака
(эмбеддинга) автомобиля для сопоставления снимков одного ТС с разных камер
**без использования государственного номера**.

## Что внутри

| Компонент | Описание |
|---|---|
| `src/` | обучение, инференс, метрики, генерация артефактов сдачи |
| `service/` | FastAPI-сервис: API, аутентификация, работа с галереей |
| `service/gallery_index.py` | индекс галереи (FAISS, эмбеддинги) — без torch |
| `service/reid_inference.py` | ансамбль моделей (инференс, attention) — с torch |
| `models/` | веса четверного ансамбля (fp16), индекс галереи, порог |
| `artifacts/` | `submission.csv`, `embeddings.npy`, `candidates.csv`, отчёты |
| `Dockerfile`, `docker-compose.yml` | запуск одной командой, офлайн |

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI (монолит)                     │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Auth         │  │ GalleryIndex │  │ ReIDInference│  │
│  │ (JWT/API key)│  │ (без torch)  │  │ (torch+GPU)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                │                  │           │
│         ▼                ▼                  ▼           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ PostgreSQL   │  │ FAISS index  │  │ PyTorch      │  │
│  │ (users,keys, │  │ gallery.npz  │  │ ensemble     │  │
│  │  watchlist,  │  │              │  │ (4 модели)   │  │
│  │  history)    │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Модульность:** `GalleryIndex` не зависит от torch — может использоваться
без GPU для работы с галереей. `ReIDInference` загружает модели ансамбля
только для инференса. Оба компонента живут в одном процессе (монолит),
но чётко разделены по ответственности.

## Быстрый старт

```bash
# веб-сервис на http://localhost:8000  (Swagger: /docs)
DATA_DIR=/путь/к/датасету docker compose up --build
```

```bash
# полный инференс на тестовой выборке -> ./artifacts/
DATA_DIR=/путь/к/датасету docker compose run --rm inference
```

`DATA_DIR` — каталог с `images/`, `train.csv`, `test_query.csv`, `test_gallery.csv`.
Интернет не требуется: веса и все зависимости — в составе образа.

Без Docker:

```bash
pip install -r requirements.txt
python -m src.run_all --data-root /путь/к/датасету --out-dir artifacts
```

## Аутентификация

Сервис поддерживает два способа аутентификации:

1. **JWT** — для веб-клиентов:
   - `POST /api/auth/register` — регистрация
   - `POST /api/auth/login` — получить JWT token
   - Использовать `Authorization: Bearer <token>`

2. **API Key** — для machine-to-machine:
   - `POST /api/auth/keys` — создать ключ (показывается один раз)
   - Использовать `X-API-Key: falcon_...`

## API

OpenAPI/Swagger: `http://localhost:8000/docs`.

### Основные endpoints (требуют аутентификации)

| Метод | Назначение |
|---|---|
| `POST /api/search` | 1–5 снимков (+ BBox) → топ-N кандидатов с уверенностью либо отказ |
| `POST /api/embed` | изображение (+ BBox) → 3840-d вектор признака |
| `POST /api/explain` | карта внимания модели (интерпретируемость) |
| `GET /api/watchlist` | список объектов «на контроле» |
| `POST /api/watchlist` | добавить в watchlist |
| `DELETE /api/watchlist/{wid}` | удалить из watchlist |
| `GET /api/stats/usage` | статистика использования |

### Публичные endpoints (без аутентификации)

| Метод | Назначение |
|---|---|
| `GET /api/health` | статус, размер галереи, активный порог |
| `GET /api/gallery/list` | пагинированный список галереи с URL |
| `GET /api/gallery/{id}/thumb` | миниатюра кандидата |
| `GET /api/gallery/{id}/frame` | полный кадр |
| `GET /api/gallery/{id}/attention` | attention-карта (precomputed) |
| `GET /api/gallery/{id}/similar` | «досье»: соседние появления объекта |
| `GET /api/stats/locations` | статистика локаций |
| `GET /api/stats/tracks` | треки ТС по локациям |
| `GET /api/val_curves` | кривые F1/TNR по порогам |

### Auth endpoints

| Метод | Назначение |
|---|---|
| `POST /api/auth/register` | регистрация пользователя |
| `POST /api/auth/login` | получение JWT token |
| `POST /api/auth/keys` | создание API key |
| `GET /api/auth/keys` | список ключей |
| `DELETE /api/auth/keys/{id}` | отзыв ключа |

## Генерация attention-карт

Для работы `/api/gallery/{id}/attention` нужно заранее сгенерировать PNG:

```bash
python -m src.build_gallery --attn
```

Сохраняет PNG в `data/attn/{id}.png`. Endpoint отдаёт готовый файл без torch.

## Структура данных

| Путь | Описание | Размер |
|------|----------|--------|
| `data/images/` | Полные кадры с камер | ~8 ГБ |
| `data/crops/` | Кропы ТС (миниатюры) | ~500 МБ |
| `data/attn/` | Attention maps (PNG) | ~500 МБ |

## Environment Variables

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `DATA_ROOT` | `/data` | Корневой каталог с данными |
| `GALLERY_INDEX` | `/app/models/gallery_index.npz` | Путь к индексу галереи |
| `REFUSAL_THRESHOLD` | `0.302` | Порог отказа |
| `DATABASE_URL` | `postgresql+asyncpg://falcon:falcon@localhost:5432/falcon` | PostgreSQL |
| `JWT_SECRET` | `change-me-in-production` | Секрет для JWT |

## Архитектура ReID

```
изображение + BBox ──► кроп (+6% контекста)
   ──► DINOv2 ViT-L/14 (336²)             ─┐
   ──► 0.7 · DINOv2 ViT-B/14 + части (252²) ┼─► конкатенация ─► 3840-d float32
   ──► 0.5 · ConvNeXt-base (224²)           ┤
   ──► DINOv3 ViT-L/16 (256²)               ─┘
   ──► DBA k=3 (галерея) + alphaQE k=1 (запрос) ─► L2-нормировка
   ──► косинусный поиск FAISS ─► топ-N | отказ (max sim < τ)
```

## Метрики (отложенная валидация, 308 неизвестных модели ТС)

| Метрика | Значение |
|---|---|
| mAP (кросс-камерный) | **0.844** |
| Rank-1 / Rank-5 | 0.812 / 0.904 |
| mINP | 0.839 |
| F1 режима отказа @ τ=0.320 | 0.768 |
| TNR (доля верных отказов) | 0.739 |

## Соответствие ограничениям

- Суммарный размер весов: 1505 МБ + 6.2 МБ детектор = 1511 МБ (< 2048 МБ) ✔
- Признаки номерных знаков не используются ✔
- Инференс офлайн, запуск одной командой ✔

## Внешние ресурсы

| Ресурс | Версия | Лицензия | Использование |
|---|---|---|---|
| PyTorch / torchvision | 2.5+ | BSD-3 | фреймворк |
| timm | 1.0.29 | Apache-2.0 | backbone-модели |
| FAISS | 1.15 | MIT | ANN-поиск |
| FastAPI / uvicorn / pydantic | 0.115+ | MIT | сервис |
| SQLAlchemy / asyncpg | 2.0+ | MIT | PostgreSQL |
| YOLOv8n (`ultralytics`) | 8.4 | AGPL-3.0 | UX-детекция ТС (6.5 МБ) |
