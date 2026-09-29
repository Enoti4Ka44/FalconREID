"""MCP-сервер для FALCON ReID API.

Оборачивает REST API в инструменты MCP для использования в AI-агентах
(Claude Desktop, Cursor, opencode и др.).

Запуск:
    python mcp_server.py                           # stdio (по умолчанию)
    python mcp_server.py --transport sse --port 8001  # SSE

Настройка:
    FALCON_API_URL=http://localhost:8000 python mcp_server.py
    FALCON_API_KEY=falcon_... python mcp_server.py
"""
import argparse
import json
import os
from pathlib import Path

import httpx
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

API_URL = os.environ.get("FALCON_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("FALCON_API_KEY", "")

server = Server("falcon-reid")

TOOLS = [
    Tool(
        name="search_vehicle",
        description="Поиск транспортного средства по изображению. "
                    "Принимает путь к файлу изображения и опциональные координаты BBox. "
                    "Возвращает топ-N кандидатов из галереи с оценками уверенности.",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Путь к файлу изображения (JPEG/PNG)",
                },
                "x": {"type": "integer", "description": "X-координата BBox (опционально)"},
                "y": {"type": "integer", "description": "Y-координата BBox (опционально)"},
                "w": {"type": "integer", "description": "Ширина BBox (опционально)"},
                "h": {"type": "integer", "description": "Высота BBox (опционально)"},
                "top_k": {
                    "type": "integer",
                    "description": "Количество кандидатов (по умолчанию 10)",
                    "default": 10,
                },
            },
            "required": ["image_path"],
        },
    ),
    Tool(
        name="batch_search",
        description="Пакетный поиск ТС: до 32 изображений, результат для каждого отдельно. "
                    "Используется для массовой идентификации и FPS-тестирования.",
        inputSchema={
            "type": "object",
            "properties": {
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Список путей к файлам изображений (до 32)",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Количество кандидатов (по умолчанию 10)",
                    "default": 10,
                },
            },
            "required": ["image_paths"],
        },
    ),
    Tool(
        name="get_gallery_list",
        description="Список объектов галереи с пагинацией. "
                    "Возвращает gallery_id, camera_group и URL-ы изображений.",
        inputSchema={
            "type": "object",
            "properties": {
                "group": {
                    "type": "integer",
                    "description": "Фильтр по камере-группе (опционально)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Смещение пагинации (по умолчанию 0)",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Количество элементов (по умолчанию 20)",
                    "default": 20,
                },
            },
        },
    ),
    Tool(
        name="get_gallery_item",
        description="Досье на объект галереи: ближайшие соседи по косинусному сходству. "
                    "Используется для проверки 'близнецов' и качества идентификации.",
        inputSchema={
            "type": "object",
            "properties": {
                "gallery_id": {
                    "type": "string",
                    "description": "Идентификатор объекта из галереи",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Количество соседей (по умолчанию 12)",
                    "default": 12,
                },
            },
            "required": ["gallery_id"],
        },
    ),
    Tool(
        name="compare_vehicles",
        description="Сравнение двух объектов галереи: косинусное сходство. "
                    "Используется для A/B-сравнения из архива.",
        inputSchema={
            "type": "object",
            "properties": {
                "gallery_id_a": {
                    "type": "string",
                    "description": "ID первого объекта",
                },
                "gallery_id_b": {
                    "type": "string",
                    "description": "ID второго объекта",
                },
            },
            "required": ["gallery_id_a", "gallery_id_b"],
        },
    ),
    Tool(
        name="get_watchlist",
        description="Список объектов 'на контроле' (watchlist). "
                    "При каждом поиске автоматически проверяется сходство с watchlist.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="add_to_watchlist",
        description="Добавить объект галереи в watchlist. "
                    "При каждом поиске будет проверяться сходство запроса с этим объектом.",
        inputSchema={
            "type": "object",
            "properties": {
                "gallery_id": {
                    "type": "string",
                    "description": "Идентификатор объекта из галереи",
                },
                "name": {
                    "type": "string",
                    "description": "Произвольное имя (опционально)",
                    "default": "",
                },
            },
            "required": ["gallery_id"],
        },
    ),
    Tool(
        name="remove_from_watchlist",
        description="Удалить объект из watchlist.",
        inputSchema={
            "type": "object",
            "properties": {
                "watchlist_id": {
                    "type": "string",
                    "description": "Идентификатор записи watchlist",
                },
            },
            "required": ["watchlist_id"],
        },
    ),
    Tool(
        name="get_stats",
        description="Статистика использования сервиса: количество запросов, "
                    "среднее время инференса, p95, доля отказов, алерты watchlist.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_my_stats",
        description="Статистика текущего пользователя: количество запросов, "
                    "время инференса, топ камеры-группы.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_my_history",
        description="История поисков текущего пользователя с топ-3 кандидатами.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Количество записей (по умолчанию 20)",
                    "default": 20,
                },
            },
        },
    ),
    Tool(
        name="get_health",
        description="Проверка здоровья сервиса: статус, устройство, размер галереи, "
                    "порог, количество моделей.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_locations",
        description="Статистика локаций: количество снимков на каждую камеру-группу.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_tracks",
        description="Треки ТС: машины, повторно замеченные в 2+ разных локациях.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


def _headers() -> dict:
    h = {}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _text(data: dict | list | str) -> list[TextContent]:
    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    return [TextContent(type="text", text=text)]


async def _handle_tool(name: str, arguments: dict) -> list[TextContent]:
    async with httpx.AsyncClient(base_url=API_URL, timeout=60.0) as client:
        try:
            if name == "search_vehicle":
                image_path = Path(arguments["image_path"])
                if not image_path.exists():
                    return _text(f"Ошибка: файл не найден: {image_path}")

                files = {"files": (image_path.name, image_path.read_bytes())}
                data = {"top_k": arguments.get("top_k", 10)}
                for k in ("x", "y", "w", "h"):
                    if k in arguments and arguments[k] is not None:
                        data[k] = arguments[k]

                resp = await client.post(
                    "/api/search",
                    files=files,
                    data=data,
                    headers=_headers(),
                )
                return _text(resp.json())

            elif name == "batch_search":
                image_paths = arguments["image_paths"]
                if len(image_paths) > 32:
                    return _text("Ошибка: не более 32 файлов")

                files = []
                for p in image_paths:
                    path = Path(p)
                    if not path.exists():
                        return _text(f"Ошибка: файл не найден: {path}")
                    files.append(("files", (path.name, path.read_bytes())))

                data = {"top_k": arguments.get("top_k", 10)}
                resp = await client.post(
                    "/api/batch_search",
                    files=files,
                    data=data,
                    headers=_headers(),
                )
                return _text(resp.json())

            elif name == "get_gallery_list":
                params = {}
                for k in ("group", "offset", "limit"):
                    if k in arguments and arguments[k] is not None:
                        params[k] = arguments[k]
                resp = await client.get("/api/gallery/list", params=params)
                return _text(resp.json())

            elif name == "get_gallery_item":
                gid = arguments["gallery_id"]
                top_k = arguments.get("top_k", 12)
                resp = await client.get(
                    f"/api/gallery/{gid}/similar",
                    params={"top_k": top_k},
                )
                return _text(resp.json())

            elif name == "compare_vehicles":
                resp = await client.get(
                    "/api/compare",
                    params={"a": arguments["gallery_id_a"], "b": arguments["gallery_id_b"]},
                )
                return _text(resp.json())

            elif name == "get_watchlist":
                resp = await client.get("/api/watchlist", headers=_headers())
                return _text(resp.json())

            elif name == "add_to_watchlist":
                resp = await client.post(
                    "/api/watchlist",
                    json={"gallery_id": arguments["gallery_id"], "name": arguments.get("name", "")},
                    headers=_headers(),
                )
                return _text(resp.json())

            elif name == "remove_from_watchlist":
                resp = await client.delete(
                    f"/api/watchlist/{arguments['watchlist_id']}",
                    headers=_headers(),
                )
                return _text(resp.json())

            elif name == "get_stats":
                resp = await client.get("/api/stats/usage", headers=_headers())
                return _text(resp.json())

            elif name == "get_my_stats":
                resp = await client.get("/api/stats/my", headers=_headers())
                return _text(resp.json())

            elif name == "get_my_history":
                limit = arguments.get("limit", 20)
                resp = await client.get(
                    "/api/stats/my/history",
                    params={"limit": limit},
                    headers=_headers(),
                )
                return _text(resp.json())

            elif name == "get_health":
                resp = await client.get("/api/health")
                return _text(resp.json())

            elif name == "get_locations":
                resp = await client.get("/api/stats/locations")
                return _text(resp.json())

            elif name == "get_tracks":
                resp = await client.get("/api/stats/tracks")
                return _text(resp.json())

            else:
                return _text(f"Неизвестный инструмент: {name}")

        except httpx.ConnectError:
            return _text(
                f"Ошибка: не удалось подключиться к {API_URL}. "
                f"Убедитесь, что FALCON API запущен."
            )
        except Exception as e:
            return _text(f"Ошибка: {e}")


async def handle_list_tools(request: ListToolsRequest) -> ListToolsResult:
    return ListToolsResult(tools=TOOLS)


async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
    name = request.params.name
    arguments = request.params.arguments or {}
    result = await _handle_tool(name, arguments)
    return CallToolResult(content=result)


server.add_request_handler("tools/list", ListToolsRequest, handle_list_tools)
server.add_request_handler("tools/call", CallToolRequest, handle_call_tool)


async def main():
    parser = argparse.ArgumentParser(description="MCP-сервер для FALCON ReID API")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Транспорт: stdio (по умолчанию) или sse",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Порт для SSE-транспорта (по умолчанию 8001)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    else:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0],
                    streams[1],
                    server.create_initialization_options(),
                )

        app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn

        print(f"MCP-сервер запущен на http://localhost:{args.port}/sse")
        await uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=args.port)
        ).serve()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
