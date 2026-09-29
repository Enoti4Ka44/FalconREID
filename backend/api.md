# FALCON ReID — описание REST API

> Сгенерировано из OpenAPI-спецификации сервиса (`docs/gen_api_md.py`).
> Интерактивная версия — Swagger UI: `http://localhost:8000/docs`,
> схема — `http://localhost:8000/openapi.json`.

## Аутентификация

Методы с пометкой 🔒 требуют один из заголовков:

```http
Authorization: Bearer <JWT>          # POST /api/auth/login
X-API-Key: falcon_<ключ>             # POST /api/auth/keys (показывается один раз)
```

Демо-пользователь локального стенда: `test` / `test`.

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" -d '{"username":"test","password":"test"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:8000/api/search -H "Authorization: Bearer $TOKEN" \
  -F "files=@frame.jpg" -F x=1202 -F y=270 -F w=588 -F h=474 -F top_k=10
```

Ошибки возвращаются как `{"detail": "..."}`: 400 — некорректный вход (не
декодируется изображение, рамка вне кадра и т.п.), 401 — нет или неверный токен,
404 — объект не найден, 422 — ошибка валидации полей.


## Методы

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/api/auth/register` | Регистрация пользователя |
| `POST` | `/api/auth/login` | Получение JWT токена |
| `GET` | `/api/auth/keys` 🔒 | Список API ключей |
| `POST` | `/api/auth/keys` 🔒 | Создание API ключа |
| `DELETE` | `/api/auth/keys/{key_id}` 🔒 | Отзыв API ключа |
| `POST` | `/api/detect` 🔒 | Детекция транспортных средств |
| `GET` | `/api/gallery/list` | Список галереи с пагинацией |
| `GET` | `/api/gallery/{image_id}/thumb` | Миниатюра кандидата |
| `GET` | `/api/gallery/{image_id}/frame` | Полный кадр |
| `GET` | `/api/gallery/{image_id}/attention` | Attention-карта кандидата |
| `GET` | `/api/gallery/{image_id}/similar` | Досье: похожие объекты |
| `GET` | `/api/health` | Проверка здоровья сервиса |
| `GET` | `/api/val_curves` | Кривые валидации |
| `POST` | `/api/search` 🔒 | Поиск транспортного средства |
| `POST` | `/api/embed` 🔒 | Извлечение эмбеддинга |
| `POST` | `/api/explain` 🔒 | Карта внимания модели |
| `POST` | `/api/batch_search` 🔒 | Пакетный поиск ТС |
| `GET` | `/api/stats/locations` | Статистика локаций |
| `GET` | `/api/stats/tracks` | Треки ТС по локациям |
| `GET` | `/api/stats/usage` 🔒 | Общая статистика использования |
| `GET` | `/api/stats/my` 🔒 | Моя статистика |
| `GET` | `/api/stats/my/history` 🔒 | Моя история поисков |
| `GET` | `/api/watchlist` 🔒 | Список watchlist |
| `POST` | `/api/watchlist` 🔒 | Добавить в watchlist |
| `DELETE` | `/api/watchlist/{wid}` 🔒 | Удалить из watchlist |
| `POST` | `/api/pair_matrix` | Матрица попарных сходств |
| `GET` | `/api/compare` | Сравнение двух объектов |

## auth

### `POST /api/auth/register` — Регистрация пользователя

Создаёт нового пользователя. Требуется уникальное имя пользователя и пароль. После регистрации можно войти через `/api/auth/login` для получения JWT токена или создать API ключ через `/api/auth/keys`.

**Тестовый пользователь:** `test` / `test` (создаётся автоматически при старте).

Тело запроса (`application/json`):

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `username` | string | да | Username |
| `password` | string | да | Password |

Ответы: `201` Successful Response → `UserResponse` · `422` Validation Error → `HTTPValidationError`

### `POST /api/auth/login` — Получение JWT токена

Аутентификация по логину и паролю. Возвращает JWT access_token, который нужно использовать в заголовке `Authorization: Bearer <token>` для доступа к защищённым эндпоинтам. Токен действителен 60 минут.

**Тестовый пользователь:**
- Логин: `test`
- Пароль: `test`

После получения токена нажмите кнопку **Authorize** вверху и введите `Bearer <полученный_токен>`.

Тело запроса (`application/json`):

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `username` | string | да | Username |
| `password` | string | да | Password |

Ответы: `200` Successful Response → `TokenResponse` · `422` Validation Error → `HTTPValidationError`

### `GET /api/auth/keys` 🔒 — Список API ключей

Возвращает список всех API ключей текущего пользователя. Сами ключи не возвращаются (только метаданные). Для доступа требуется JWT токен.

Ответы: `200` Successful Response → `list[APIKeyInfo]`

### `POST /api/auth/keys` 🔒 — Создание API ключа

Создаёт новый API ключ для machine-to-machine интеграций. Ключ показывается **только один раз** при создании — сохраните его. Используйте ключ в заголовке `X-API-Key: falcon_...` для доступа к API. Для доступа требуется JWT токен.

Тело запроса (`application/json`):

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `name` | string | нет | Name |

Ответы: `201` Successful Response → `APIKeyResponse` · `422` Validation Error → `HTTPValidationError`

### `DELETE /api/auth/keys/{key_id}` 🔒 — Отзыв API ключа

Деактивирует API ключ. После отзыва ключ нельзя использовать для доступа к API. Операция необратима. Для доступа требуется JWT токен.

| Параметр | Где | Тип | Обяз. | Описание |
|---|---|---|---|---|
| `key_id` | path | string | да |  |

Ответы: `204` Successful Response · `422` Validation Error → `HTTPValidationError`

## detect

### `POST /api/detect` 🔒 — Детекция транспортных средств

Принимает изображение и возвращает bounding box'ы обнаруженных транспортных средств с использованием YOLOv8n.

**Используется для:**
- UX-подсветки ТС на кадре в интерфейсе
- Автоматического определения BBox для поиска

**Важно:** По ТЗ детекция НЕ входит в задачу оценки. Официальные артефакты считаются по выданным BBox, детектор — только удобство интерфейса.

**Параметры:**
- `file` — JPEG/PNG изображение (полный кадр с камеры)

**Ответ:**
- `width, height` — размеры изображения
- `boxes` — список обнаруженных ТС с координатами, score и классом

Тело запроса (`multipart/form-data`):

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `file` | string | да | Изображение с камеры |

Ответы: `200` Successful Response → `DetectResponse` · `422` Validation Error → `HTTPValidationError`

## gallery

### `GET /api/gallery/list` — Список галереи с пагинацией

Возвращает пагинированный список объектов галереи. Каждый элемент содержит gallery_id, camera_group и URL-ы для получения миниатюры, полного кадра и attention-карты.

**Параметры:**
- `group` — фильтр по камере-группе (опционально)
- `offset` — смещение (по умолчанию 0)
- `limit` — количество элементов (по умолчанию 60, макс 200)

**Ответ:**
- `total` — общее количество элементов
- `offset` — текущее смещение
- `items` — список объектов с URL-ами к ресурсам

| Параметр | Где | Тип | Обяз. | Описание |
|---|---|---|---|---|
| `group` | query | integer | null | нет | Фильтр по камере-группе |
| `offset` | query | integer | нет | Смещение пагинации |
| `limit` | query | integer | нет | Количество элементов (макс 200) |

Ответы: `200` Successful Response → `GalleryPage` · `422` Validation Error → `HTTPValidationError`

### `GET /api/gallery/{image_id}/thumb` — Миниатюра кандидата

Возвращает JPEG миниатюру (кроп) транспортного средства из галереи. Используется для отображения в карточках кандидатов.

**Параметры:**
- `image_id` — идентификатор изображения из галереи

**Ответ:** JPEG изображение

| Параметр | Где | Тип | Обяз. | Описание |
|---|---|---|---|---|
| `image_id` | path | string | да |  |

Ответы: `200` Successful Response · `404` Миниатюра не найдена · `422` Validation Error → `HTTPValidationError`

### `GET /api/gallery/{image_id}/frame` — Полный кадр

Возвращает полный кадр (исходное изображение с камеры) для указанного объекта галереи. Используется для детального просмотра.

**Параметры:**
- `image_id` — идентификатор изображения из галереи

**Ответ:** JPEG изображение полного кадра

| Параметр | Где | Тип | Обяз. | Описание |
|---|---|---|---|---|
| `image_id` | path | string | да |  |

Ответы: `200` Successful Response · `404` Кадр не найден · `422` Validation Error → `HTTPValidationError`

### `GET /api/gallery/{image_id}/attention` — Attention-карта кандидата

Возвращает precomputed attention-карту (heatmap) для объекта галереи в формате PNG. Карта показывает, на какие области изображения обращает внимание нейросеть.

**Важно:** Для работы необходимо заранее сгенерировать карты:
```
python -m src.build_gallery --attn
```

**Параметры:**
- `image_id` — идентификатор изображения из галереи

**Ответ:** PNG изображение attention-карты

| Параметр | Где | Тип | Обяз. | Описание |
|---|---|---|---|---|
| `image_id` | path | string | да |  |

Ответы: `200` Successful Response · `404` Attention-карта не сгенерирована · `422` Validation Error → `HTTPValidationError`

### `GET /api/gallery/{image_id}/similar` — Досье: похожие объекты

Возвращает список ближайших соседей объекта в галерее по косинусному сходству эмбеддингов. Используется для:
- Просмотра «досье» ТС — где ещё появлялась эта машина
- Нахождения «близнецов» — похожих, но разных ТС
- Проверки качества идентификации

**Параметры:**
- `image_id` — идентификатор изображения из галереи
- `top_k` — количество соседей (по умолчанию 12)

**Ответ:**
- `gallery_id` — исходный объект
- `camera_group` — камера-группа исходного объекта
- `neighbors` — список соседей с confidence и same_scene

| Параметр | Где | Тип | Обяз. | Описание |
|---|---|---|---|---|
| `image_id` | path | string | да |  |
| `top_k` | query | integer | нет | Количество соседей |

Ответы: `200` Successful Response → `GallerySimilar` · `404` Объект не найден в галерее · `422` Validation Error → `HTTPValidationError`

## health

### `GET /api/health` — Проверка здоровья сервиса

Возвращает текущий статус сервиса: состояние, размер галереи, активный порог отказа, количество моделей в ансамбле.

**Используется для:**
- Мониторинга доступности сервиса
- Проверки конфигурации перед запросами
- Healthcheck в Docker/Kubernetes

**Не требует аутентификации.**

Ответы: `200` Successful Response → `HealthResponse`

### `GET /api/val_curves` — Кривые валидации

Возвращает кривые F1, TNR, Precision, Recall по сетке порогов для визуализации в интерфейсе.

**Используется для:**
- Умного слайдера порога в UI
- Выбора оптимального порога отказа
- Обоснования выбора порога на защите

**Ответ:**
- `grid` — сетка значений порога
- `F1, TNR, precision, recall` — значения метрик

**Не требует аутентификации.**

Ответы: `200` Successful Response → `object` · `404` Кривые валидации не найдены в модели

## search

### `POST /api/search` 🔒 — Поиск транспортного средства

Принимает 1–5 изображений одного ТС (с BBox) и возвращает топ-N кандидатов из галереи с оценками уверенности, либо отказ если уверенного совпадения нет.

**Параметры:**
- `files` — 1–5 JPEG/PNG изображений одного ТС
- `x, y, w, h` — координаты BBox (применяется к первому изображению)
- `top_k` — количество кандидатов (по умолчанию 10)
- `threshold` — переопределить порог отказа

**Ответ содержит:**
- `candidates` — список кандидатов с gallery_id, confidence, accepted
- `refused` — true если ни один кандидат не прошёл порог
- `inference_ms` — время инференса в миллисекундах
- `sim_hist` — гистограмма близостей (для UI)
- `watchlist_alerts` — совпадения с watchlist

Тело запроса (`multipart/form-data`):

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `files` | list[string] | да | 1–5 снимков одного ТС |
| `x` | integer | null | нет | X-координата левого верхнего угла BBox |
| `y` | integer | null | нет | Y-координата левого верхнего угла BBox |
| `w` | integer | null | нет | Ширина BBox в пикселях |
| `h` | integer | null | нет | Высота BBox в пикселях |
| `top_k` | integer | нет | Количество кандидатов для возврата (1–100) |
| `threshold` | number | null | нет | Переопределить порог отказа (0–1) |

Ответы: `200` Successful Response → `SearchResponse` · `422` Validation Error → `HTTPValidationError`

### `POST /api/embed` 🔒 — Извлечение эмбеддинга

Принимает изображение ТС (с опциональным BBox) и возвращает L2-нормированный float32 вектор признака (3072-d для текущего ансамбля). Используется для кастомной постобработки или интеграций.

**Параметры:**
- `file` — JPEG/PNG изображение
- `x, y, w, h` — координаты BBox (опционально)

**Ответ:**
- `dim` — размерность вектора (3072 для текущего ансамбля)
- `embedding` — массив float значений

Тело запроса (`multipart/form-data`):

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `file` | string | да | Изображение ТС |
| `x` | integer | null | нет | X-координата BBox |
| `y` | integer | null | нет | Y-координата BBox |
| `w` | integer | null | нет | Ширина BBox |
| `h` | integer | null | нет | Высота BBox |

Ответы: `200` Successful Response → `EmbedResponse` · `422` Validation Error → `HTTPValidationError`

### `POST /api/explain` 🔒 — Карта внимания модели

Принимает изображение ТС и возвращает attention-карту (heatmap) в формате base64 PNG. Показывает, на какие области изображения обращает внимание нейросеть при извлечении признаков.

**Используется для:**
- Интерпретируемости решения
- Проверки, что модель не опирается на номерные знаки
- Отладки и анализа

**Параметры:**
- `file` — JPEG/PNG изображение
- `x, y, w, h` — координаты BBox (опционально)

**Ответ:**
- `image_base64` — PNG изображение attention-карты в base64

Тело запроса (`multipart/form-data`):

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `file` | string | да | Изображение ТС |
| `x` | integer | null | нет | X-координата BBox |
| `y` | integer | null | нет | Y-координата BBox |
| `w` | integer | null | нет | Ширина BBox |
| `h` | integer | null | нет | Высота BBox |

Ответы: `200` Successful Response → `object` · `422` Validation Error → `HTTPValidationError`

### `POST /api/batch_search` 🔒 — Пакетный поиск ТС

Принимает массив изображений (до 32) и возвращает результат поиска для каждого изображения отдельно.

**Используется для:**
- Пакетной обработки видеопотока
- FPS-тестирования производительности
- Массовой идентификации ТС

**Параметры:**
- `files` — до 32 JPEG/PNG изображений ТС
- `top_k` — количество кандидатов (по умолчанию 10)
- `threshold` — переопределить порог отказа

**Ответ:** массив SearchResponse (по одному на файл)

**Пример ответа:**
```json
[
  {
    "query_embedding_dim": 3072,
    "inference_ms": 42.3,
    "threshold": 0.455,
    "refused": false,
    "message": "Принято кандидатов: 5",
    "n_query_images": 1,
    "sim_max": 0.85,
    "sim_hist": {"0.0-0.1": 0, "0.1-0.2": 2},
    "candidates": [
      {"gallery_id": "abc", "confidence": 0.85, "accepted": true, "camera_group": 1}
    ],
    "watchlist_alerts": []
  }
]
```

**Требует аутентификации.**

Тело запроса (`multipart/form-data`):

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `files` | list[string] | да | До 32 изображений ТС |
| `top_k` | integer | нет | Количество кандидатов (1–100) |
| `threshold` | number | null | нет | Порог отказа (0–1) |

Ответы: `200` Successful Response → `list[SearchResponse]` · `422` Validation Error → `HTTPValidationError`

## stats

### `GET /api/stats/locations` — Статистика локаций

Возвращает количество снимков на каждую камеру-группу. Камеры-группы восстановлены по фону кадра и геометрии BBox без какой-либо разметки.

**Используется для:**
- Визуализации покрытия камер на карте
- Фильтрации галереи по локациям
- Анализа распределения данных

**Ответ:**
- `locations` — список {camera_group, count}
- `total` — общее количество объектов в галерее

**Не требует аутентификации.**

Ответы: `200` Successful Response → `object`

### `GET /api/stats/tracks` — Треки ТС по локациям

Возвращает «треки» — связные компоненты галереи по косинусному сходству >= порога, замеченные в 2+ разных локациях. Это машины, которые повторно появлялись в разных частях города.

**Используется для:**
- Поиска «перемещающихся» ТС
- Расследования маршрутов
- Визуализации на карте

**Ответ:**
- `threshold` — порог сходства
- `n_tracks` — количество треков
- `tracks` — список треков с участниками и локациями

**Не требует аутентификации.**

Ответы: `200` Successful Response → `object`

### `GET /api/stats/usage` 🔒 — Общая статистика использования

Возвращает агрегированную статистику по всем поисковым запросам: количество запросов, среднее время инференса, p95, доля отказов, количество алертов watchlist.

**Ответ:**
- `requests` — общее количество запросов
- `avg_ms, p95_ms` — время инференса
- `refusal_rate` — доля отказов
- `alerts_total` — количество алертов
- `recent` — последние 40 записей

**Требует аутентификации.**

Ответы: `200` Successful Response → `UsageStats`

### `GET /api/stats/my` 🔒 — Моя статистика

Возвращает статистику поисковых запросов текущего пользователя: количество запросов, среднее время инференса, p95, доля отказов, количество алертов, топ камеры-группы.

**Ответ:**
- `requests` — количество запросов пользователя
- `avg_ms, p95_ms` — время инференса
- `refusal_rate` — доля отказов
- `alerts_total` — количество алертов
- `top_cameras` — топ камеры-групп по частоте появления в результатах

**Требует аутентификации.**

Ответы: `200` Successful Response → `UserStats`

### `GET /api/stats/my/history` 🔒 — Моя история поисков

Возвращает последние поисковые запросы текущего пользователя с топ-3 кандидатами в каждом результате.

**Параметры:**
- `limit` — количество записей (по умолчанию 20, макс 100)

**Ответ (каждый элемент):**
- `id` — идентификатор запроса
- `refused` — был ли отказ
- `n_accepted` — количество принятых кандидатов
- `inference_ms` — время инференса
- `top3` — топ-3 кандидата (gallery_id, confidence, camera_group)
- `created_at` — время запроса (Unix timestamp)

**Требует аутентификации.**

| Параметр | Где | Тип | Обяз. | Описание |
|---|---|---|---|---|
| `limit` | query | integer | нет | Количество записей (макс 100) |

Ответы: `200` Successful Response → `list[HistoryItem]` · `422` Validation Error → `HTTPValidationError`

## watchlist

### `GET /api/watchlist` 🔒 — Список watchlist

Возвращает все объекты текущего пользователя в watchlist («на контроле»). При каждом поиске автоматически проверяется сходство запроса с объектами watchlist.

**Используется для:**
- Мониторинга разыскиваемых ТС
- Получения алертов при совпадении

**Ответ:**
- `items` — список объектов с id, name, gallery_id, created

**Требует аутентификации.**

Ответы: `200` Successful Response → `object`

### `POST /api/watchlist` 🔒 — Добавить в watchlist

Добавляет объект галереи в watchlist текущего пользователя. При каждом поиске будет автоматически проверяться сходство запроса с этим объектом.

**Параметры:**
- `gallery_id` — идентификатор объекта из галереи
- `name` — произвольное имя (опционально)

**Ответ:**
- `id` — идентификатор записи watchlist
- `gallery_id` — подтверждение добавленного объекта

**Требует аутентификации.**

Тело запроса (`application/json`):

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `gallery_id` | string | да | Gallery Id |
| `name` | string | нет | Name |

Ответы: `200` Successful Response → `object` · `404` Объект не найден в галерее · `422` Validation Error → `HTTPValidationError`

### `DELETE /api/watchlist/{wid}` 🔒 — Удалить из watchlist

Удаляет объект из watchlist текущего пользователя.

**Параметры:**
- `wid` — идентификатор записи watchlist

**Требует аутентификации.**

| Параметр | Где | Тип | Обяз. | Описание |
|---|---|---|---|---|
| `wid` | path | string | да |  |

Ответы: `200` Successful Response → `object` · `404` Запись не найдена · `422` Validation Error → `HTTPValidationError`

### `POST /api/pair_matrix` — Матрица попарных сходств

Возвращает матрицу попарных косинусных сходств для списка объектов галереи. Используется для визуализации сходства между кандидатами.

**Параметры:**
- `ids` — список gallery_id (от 2 до 30)

**Ответ:**
- `ids` — исходный список
- `matrix` — матрица NxN сходств

**Требует аутентификации.**

Тело запроса (`application/json`):

`list[string]`

Ответы: `200` Successful Response → `object` · `400` Количество ids вне диапазона 2..30 · `404` Один из gallery_id не найден · `422` Validation Error → `HTTPValidationError`

### `GET /api/compare` — Сравнение двух объектов

Возвращает косинусное сходство между двумя объектами галереи.

**Параметры:**
- `a` — gallery_id первого объекта
- `b` — gallery_id второго объекта

**Ответ:**
- `a, b` — исходные идентификаторы
- `confidence` — косинусное сходство (0..1)

**Требует аутентификации.**

| Параметр | Где | Тип | Обяз. | Описание |
|---|---|---|---|---|
| `a` | query | string | да |  |
| `b` | query | string | да |  |

Ответы: `200` Successful Response → `object` · `404` Один из gallery_id не найден · `422` Validation Error → `HTTPValidationError`

## Схемы ответов

### `APIKeyCreate`

| Поле | Тип | Описание |
|---|---|---|
| `name` | string |  |

### `APIKeyInfo`

| Поле | Тип | Описание |
|---|---|---|
| `id` | string |  |
| `name` | string |  |
| `is_active` | boolean |  |
| `created_at` | string |  |

### `APIKeyResponse`

| Поле | Тип | Описание |
|---|---|---|
| `id` | string |  |
| `name` | string |  |
| `key` | string |  |
| `created_at` | string |  |

### `BBox`

| Поле | Тип | Описание |
|---|---|---|
| `x` | integer |  |
| `y` | integer |  |
| `w` | integer |  |
| `h` | integer |  |
| `score` | number |  |
| `cls` | string |  |

### `DetectResponse`

| Поле | Тип | Описание |
|---|---|---|
| `width` | integer |  |
| `height` | integer |  |
| `boxes` | list[BBox] |  |

### `EmbedResponse`

| Поле | Тип | Описание |
|---|---|---|
| `dim` | integer |  |
| `embedding` | list[number] |  |

### `GalleryItem`

| Поле | Тип | Описание |
|---|---|---|
| `gallery_id` | string |  |
| `camera_group` | integer |  |
| `thumb_url` | string | null |  |
| `frame_url` | string | null |  |
| `attention_url` | string | null |  |

### `GalleryPage`

| Поле | Тип | Описание |
|---|---|---|
| `total` | integer |  |
| `offset` | integer |  |
| `items` | list[GalleryItem] |  |

### `GallerySimilar`

| Поле | Тип | Описание |
|---|---|---|
| `gallery_id` | string |  |
| `camera_group` | integer |  |
| `neighbors` | list[Neighbor] |  |

### `HealthResponse`

| Поле | Тип | Описание |
|---|---|---|
| `status` | string |  |
| `device` | string |  |
| `gallery_size` | integer |  |
| `threshold` | number |  |
| `models` | integer |  |

### `HistoryItem`

| Поле | Тип | Описание |
|---|---|---|
| `id` | string |  |
| `refused` | boolean |  |
| `n_accepted` | integer |  |
| `inference_ms` | number |  |
| `n_images` | integer |  |
| `alerts` | integer |  |
| `top3` | list[Top3Candidate] |  |
| `created_at` | number |  |

### `Neighbor`

| Поле | Тип | Описание |
|---|---|---|
| `gallery_id` | string |  |
| `confidence` | number |  |
| `camera_group` | integer |  |
| `same_scene` | boolean |  |

### `SearchResponse`

| Поле | Тип | Описание |
|---|---|---|
| `query_embedding_dim` | integer |  |
| `inference_ms` | number |  |
| `threshold` | number |  |
| `refused` | boolean |  |
| `message` | string |  |
| `n_query_images` | integer |  |
| `sim_max` | number |  |
| `sim_hist` | object |  |
| `candidates` | list[SearchResult] |  |
| `watchlist_alerts` | list[object] |  |

### `SearchResult`

| Поле | Тип | Описание |
|---|---|---|
| `gallery_id` | string |  |
| `confidence` | number |  |
| `accepted` | boolean |  |
| `camera_group` | integer |  |

### `TokenResponse`

| Поле | Тип | Описание |
|---|---|---|
| `access_token` | string |  |
| `token_type` | string |  |

### `Top3Candidate`

| Поле | Тип | Описание |
|---|---|---|
| `gallery_id` | string |  |
| `confidence` | number |  |
| `camera_group` | integer |  |

### `TopCamera`

| Поле | Тип | Описание |
|---|---|---|
| `camera_group` | integer |  |
| `count` | integer |  |

### `UsageStats`

| Поле | Тип | Описание |
|---|---|---|
| `requests` | integer |  |
| `avg_ms` | number | null |  |
| `p95_ms` | number | null |  |
| `refusal_rate` | number | null |  |
| `alerts_total` | integer |  |
| `recent` | list[object] |  |

### `UserLogin`

| Поле | Тип | Описание |
|---|---|---|
| `username` | string |  |
| `password` | string |  |

### `UserRegister`

| Поле | Тип | Описание |
|---|---|---|
| `username` | string |  |
| `password` | string |  |

### `UserResponse`

| Поле | Тип | Описание |
|---|---|---|
| `id` | string |  |
| `username` | string |  |
| `created_at` | string |  |

### `UserStats`

| Поле | Тип | Описание |
|---|---|---|
| `requests` | integer |  |
| `avg_ms` | number | null |  |
| `p95_ms` | number | null |  |
| `refusal_rate` | number | null |  |
| `alerts_total` | integer |  |
| `top_cameras` | list[TopCamera] |  |

### `WatchAdd`

| Поле | Тип | Описание |
|---|---|---|
| `gallery_id` | string |  |
| `name` | string |  |
