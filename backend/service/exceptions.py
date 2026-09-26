"""Кастомные исключения и обработчики ошибок."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail


class NotFoundError(AppError):
    def __init__(self, detail: str = "not found"):
        super().__init__(404, detail)


class BadRequestError(AppError):
    def __init__(self, detail: str = "bad request"):
        super().__init__(400, detail)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})
