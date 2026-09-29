"""Поиск кадров (JPEG/PNG, ТЗ §5) и валидация входа API."""
import pytest
from pydantic import ValidationError

from src.config import find_image


def test_find_image_prefers_jpg_then_png(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    assert find_image(tmp_path, "a").name == "a.png"
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert find_image(tmp_path, "a").name == "a.jpg"
    assert find_image(tmp_path, "missing").name == "missing.jpg"


def test_register_schema_limits():
    from service.schemas.auth import UserRegister
    UserRegister(username="falcon.operator", password="12345678")
    for bad in ({"username": "ab", "password": "12345678"},
                {"username": "x" * 65, "password": "12345678"},
                {"username": "ok_user", "password": "short"},
                {"username": "bad name", "password": "12345678"}):
        with pytest.raises(ValidationError):
            UserRegister(**bad)
