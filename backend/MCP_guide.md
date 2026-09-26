# FALCON ReID — MCP-сервер: подключение и использование

## Что это

**MCP (Model Context Protocol)** — протокол, который позволяет AI-агентам
(Claude, Cursor, opencode) использовать FALCON API как «инструменты».
Агент получает доступ к поиску ТС, галерее, вочлисту и статистике.

```
┌─────────────┐     MCP      ┌──────────────┐     HTTP      ┌──────────────┐
│ Claude /    │ ◄──────────► │ mcp_server.py│ ◄───────────► │ FALCON API   │
│ Cursor /    │   (stdio     │              │   (localhost   │ (FastAPI)    │
│ opencode    │    или SSE)  │  14 tools    │    :8000)     │              │
└─────────────┘              └──────────────┘              └──────────────┘
```

## Быстрый старт

###1. Запустить FALCON API

```bash
docker compose up --build
```

API будет доступен на `http://localhost:8000`.

###2. Получить API ключ

```bash
# Получить JWT токен
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "test"}' | jq -r '.access_token')

# Создать API ключ
curl -s -X POST http://localhost:8000/api/auth/keys \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"name": "mcp-key"}' | jq -r '.key'

# Ответ: falcon_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

###3. Настроить AI-агент

Выберите свой агент и следуйте инструкции ниже.

---

## Настройка Claude Desktop

### macOS

Файл: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "falcon-reid": {
      "command": "python",
      "args": ["/home/ivan/PycharmProjects/MoscowHack/mcp_server.py"],
      "env": {
        "FALCON_API_URL": "http://localhost:8000",
        "FALCON_API_KEY": "falcon_ваш_ключ"
      }
    }
  }
}
```

### Windows

Файл: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "falcon-reid": {
      "command": "python",
      "args": ["C:\\Users\\Имя\\PycharmProjects\\MoscowHack\\mcp_server.py"],
      "env": {
        "FALCON_API_URL": "http://localhost:8000",
        "FALCON_API_KEY": "falcon_ваш_ключ"
      }
    }
  }
}
```

### После настройки

1. Перезапустить Claude Desktop
2. В диалоге появится иконка 🔧 (инструменты)
3. Спросить: «Найди машину на фото /path/to/image.jpg»

---

## Настройка opencode

Файл: `~/.config/opencode/config.json`

```json
{
  "mcpServers": {
    "falcon-reid": {
      "command": "python",
      "args": ["/home/ivan/PycharmProjects/MoscowHack/mcp_server.py"],
      "env": {
        "FALCON_API_URL": "http://localhost:8000",
        "FALCON_API_KEY": "falcon_ваш_ключ"
      }
    }
  }
}
```

---

## Настройка Cursor

Файл: `.cursor/mcp.json` в корне проекта

```json
{
  "mcpServers": {
    "falcon-reid": {
      "command": "python",
      "args": ["/home/ivan/PycharmProjects/MoscowHack/mcp_server.py"],
      "env": {
        "FALCON_API_URL": "http://localhost:8000",
        "FALCON_API_KEY": "falcon_ваш_ключ"
      }
    }
  }
}
```

---

## Доступные инструменты

### Поиск

| Инструмент | Описание | Параметры |
|------------|----------|-----------|
| `search_vehicle` | Поиск ТС по изображению | `image_path`, `x/y/w/h` (опционально), `top_k` |
| `batch_search` | Пакетный поиск (до 32 файлов) | `image_paths[]`, `top_k` |

### Галерея

| Инструмент | Описание | Параметры |
|------------|----------|-----------|
| `get_gallery_list` | Список галереи | `group`, `offset`, `limit` |
| `get_gallery_item` | Досье на объект | `gallery_id`, `top_k` |
| `compare_vehicles` | Сравнение двух объектов | `gallery_id_a`, `gallery_id_b` |

### Watchlist

| Инструмент | Описание | Параметры |
|------------|----------|-----------|
| `get_watchlist` | Список watchlist | — |
| `add_to_watchlist` | Добавить в watchlist | `gallery_id`, `name` |
| `remove_from_watchlist` | Удалить из watchlist | `watchlist_id` |

### Статистика

| Инструмент | Описание | Параметры |
|------------|----------|-----------|
| `get_stats` | Общая статистика | — |
| `get_my_stats` | Статистика пользователя | — |
| `get_my_history` | История поисков | `limit` |
| `get_health` | Проверка здоровья | — |
| `get_locations` | Статистика локаций | — |
| `get_tracks` | Треки ТС | — |

---

## Примеры использования

### Поиск ТС

```
Ты: Найди машину на фото /tmp/car1.jpg
Claude: [вызывает search_vehicle(image_path="/tmp/car1.jpg")]
Claude: Найдено 5 кандидатов:
  1. gallery_id: abc123, уверенность: 0.85, камера: 1
  2. gallery_id: def456, уверенность: 0.72, камера: 2
  ...
```

### Сравнение двух машин

```
Ты: Сравни машины abc123 и def456
Claude: [вызывает compare_vehicles(gallery_id_a="abc123", gallery_id_b="def456")]
Claude: Косинусное сходство: 0.82. Машины很可能 одинаковые.
```

### Пакетная обработка

```
Ты: Проверь эти 5 фото: /tmp/1.jpg, /tmp/2.jpg, /tmp/3.jpg, /tmp/4.jpg, /tmp/5.jpg
Claude: [вызывает batch_search(image_paths=["/tmp/1.jpg", ...])]
Claude: Результаты:
  1. /tmp/1.jpg → abc123 (0.85)
  2. /tmp/2.jpg → не найдено (отказ)
  ...
```

### Добавление в watchlist

```
Ты: Поставь машину abc123 на контроль под именем "подозреваемая"
Claude: [вызывает add_to_watchlist(gallery_id="abc123", name="подозреваемая")]
Claude: Машина abc123 добавлена в watchlist. Теперь каждый поиск будет проверять сходство.
```

### Просмотр статистики

```
Ты: Покажи статистику использования
Claude: [вызывает get_stats()]
Claude: Статистика:
  - Запросов: 42
  - Среднее время: 38.5 мс
  - P95: 65.2 мс
  - Доля отказов: 12%
  - Алертов: 3
```

---

## Транспорты

| Транспорт | Когда использовать | Запуск |
|-----------|-------------------|--------|
| **stdio** | Claude Desktop, Cursor, opencode | `python mcp_server.py` |
| **SSE** | Веб-клиенты, удаленное подключение | `python mcp_server.py --transport sse --port 8001` |

### SSE-транспорт

Для веб-клиентов или удаленного подключения:

```bash
python mcp_server.py --transport sse --port 8001
```

Подключение: `http://localhost:8001/sse`

---

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `FALCON_API_URL` | `http://localhost:8000` | URL FALCON API |
| `FALCON_API_KEY` | `""` | API ключ (опционально) |

---

## Запуск без Docker

Если FALCON API запущен локально (без Docker):

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить API
uvicorn service.app:app --host 0.0.0.0 --port 8000

# В другом терминале — MCP-сервер
python mcp_server.py
```

---

## Запуск в Docker

MCP-сервер можно запустить внутри Docker-контейнера:

```yaml
# docker-compose.yml
services:
  mcp:
    build: .
    command: python mcp_server.py --transport sse --port 8001
    ports:
      - "8001:8001"
    environment:
      FALCON_API_URL: http://api:8000
      FALCON_API_KEY: ${FALCON_API_KEY}
    depends_on:
      - api
```

---

## Диагностика

### Claude не видит инструменты

1. Проверить путь к `mcp_server.py` в конфиге
2. Проверить, что Python доступен в PATH
3. Перезапустить Claude Desktop

### Ошибка подключения

```bash
# Проверить, что API запущен
curl http://localhost:8000/api/health

# Должно вернуть:
# {"status":"ok","device":"cpu","gallery_size":...,"threshold":0.455,"models":2}
```

### Нет прав на чтение файлов

MCP-сервер читает файлы от имени пользователя, запустившего Claude/opencode.
Убедитесь, что файлы доступны для чтения.

### SSE не работает

```bash
# Проверить порт
netstat -tlnp | grep 8001

# Проверить логи
python mcp_server.py --transport sse --port 8001
```

---

## Архитектура

```
mcp_server.py
├──14 инструментов (Tools)
├── Транспорт: stdio или SSE
├── Клиент: httpx → FALCON API
└── Конфиг: переменные окружения

FALCON API (FastAPI)
├── /api/search — поиск ТС
├── /api/batch_search — пакетный поиск
├── /api/gallery/* — галерея
├── /api/watchlist — вочлист
├── /api/stats/* — статистика
└── /api/health — здоровье
```

---

## Безопасность

- API ключ передается в заголовке `X-API-Key`
- MCP-сервер не хранит ключи — они берутся из переменных окружения
- Для production используйте HTTPS и валидные ключи
- Не передавайте ключи в логах или чатах

---

## Ссылки

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Claude Desktop MCP](https://docs.anthropic.com/claude/docs/model-context-protocol)
- [FALCON ReID API](http://localhost:8000/docs) (Swagger UI)
