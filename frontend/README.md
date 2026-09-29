# FALCON ReID Landing

Одностраничный React-лендинг сервиса визуального поиска автомобилей. Интерфейс собран на shadcn/ui, Radix UI и Tailwind CSS.

## Локальный запуск

```bash
npm install
npm run dev
```

Production-сборка:

```bash
npm run build
npm run preview
```

## Docker

```bash
docker build -t falcon-reid-landing .
docker run --rm -p 8080:80 falcon-reid-landing
```

Ссылки `/app`, `/docs` и `/openapi.json` рассчитаны на размещение лендинга на одном домене с основным FastAPI-сервисом FALCON ReID.
