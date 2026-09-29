# -*- coding: utf-8 -*-
"""Генерация api.md из OpenAPI-спецификации работающего сервиса.

    python docs/gen_api_md.py --url http://localhost:8000/openapi.json --out backend/api.md

Документ собирается из той же схемы, что показывает Swagger UI, поэтому
описание API всегда совпадает с кодом.
"""
import argparse
import json
import urllib.request

HEADER = """# FALCON ReID — описание REST API

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
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \\
  -H "Content-Type: application/json" -d '{"username":"test","password":"test"}' \\
  | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:8000/api/search -H "Authorization: Bearer $TOKEN" \\
  -F "files=@frame.jpg" -F x=1202 -F y=270 -F w=588 -F h=474 -F top_k=10
```

Ошибки возвращаются как `{"detail": "..."}`: 400 — некорректный вход (не
декодируется изображение, рамка вне кадра и т.п.), 401 — нет или неверный токен,
404 — объект не найден, 422 — ошибка валидации полей.

"""


def ref_name(schema):
    if "$ref" in schema:
        return schema["$ref"].split("/")[-1]
    if schema.get("type") == "array":
        return f"list[{ref_name(schema.get('items', {}))}]"
    if "anyOf" in schema:
        return " | ".join(ref_name(s) for s in schema["anyOf"])
    return schema.get("type", "object")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/openapi.json")
    ap.add_argument("--out", default="backend/api.md")
    a = ap.parse_args()
    spec = json.loads(urllib.request.urlopen(a.url).read())
    comps = spec.get("components", {}).get("schemas", {})
    out = [HEADER]
    by_tag = {}
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            by_tag.setdefault((op.get("tags") or ["other"])[0], []).append((path, method, op))
    out.append("## Методы\n")
    out.append("| Метод | Путь | Назначение |\n|---|---|---|")
    for tag, ops in by_tag.items():
        for path, method, op in ops:
            lock = " 🔒" if op.get("security") else ""
            out.append(f"| `{method.upper()}` | `{path}`{lock} | {op.get('summary', '')} |")
    out.append("")
    for tag, ops in by_tag.items():
        out.append(f"## {tag}\n")
        for path, method, op in ops:
            lock = " 🔒" if op.get("security") else ""
            out.append(f"### `{method.upper()} {path}`{lock} — {op.get('summary', '')}\n")
            if op.get("description"):
                out.append(op["description"].strip() + "\n")
            params = op.get("parameters", [])
            if params:
                out.append("| Параметр | Где | Тип | Обяз. | Описание |\n|---|---|---|---|---|")
                for p in params:
                    out.append(f"| `{p['name']}` | {p['in']} | {ref_name(p.get('schema', {}))} | "
                               f"{'да' if p.get('required') else 'нет'} | {p.get('description', '')} |")
                out.append("")
            body = op.get("requestBody", {}).get("content", {})
            for ctype, c in body.items():
                sch = c.get("schema", {})
                name = ref_name(sch)
                props = comps.get(name, {}).get("properties", {}) if "$ref" in sch else sch.get("properties", {})
                req = set(comps.get(name, {}).get("required", [])) if "$ref" in sch else set()
                out.append(f"Тело запроса (`{ctype}`):\n")
                if props:
                    out.append("| Поле | Тип | Обяз. | Описание |\n|---|---|---|---|")
                    for k, v in props.items():
                        out.append(f"| `{k}` | {ref_name(v)} | {'да' if k in req else 'нет'} | "
                                   f"{v.get('description', v.get('title', ''))} |")
                else:
                    out.append(f"`{name}`")
                out.append("")
            resp = []
            for code, r in op.get("responses", {}).items():
                sch = r.get("content", {}).get("application/json", {}).get("schema")
                resp.append(f"`{code}` {r.get('description', '')}"
                            + (f" → `{ref_name(sch)}`" if sch else ""))
            out.append("Ответы: " + " · ".join(resp) + "\n")
    out.append("## Схемы ответов\n")
    for name, sch in comps.items():
        if name.startswith(("Body_", "HTTPValidationError", "ValidationError")):
            continue
        props = sch.get("properties", {})
        if not props:
            continue
        out.append(f"### `{name}`\n")
        out.append("| Поле | Тип | Описание |\n|---|---|---|")
        for k, v in props.items():
            out.append(f"| `{k}` | {ref_name(v)} | {v.get('description', '')} |")
        out.append("")
    open(a.out, "w", encoding="utf-8").write("\n".join(out))
    print("saved:", a.out)


if __name__ == "__main__":
    main()
