"""Pydantic-схемы для аутентификации."""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class UserRegister(BaseModel):
    # те же ограничения, что и в форме регистрации интерфейса
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-zА-Яа-яЁё0-9._-]+$")
    # bcrypt учитывает только первые 72 байта пароля
    password: str = Field(min_length=8, max_length=72)


class UserLogin(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: UUID
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}


class APIKeyCreate(BaseModel):
    name: str = ""


class APIKeyResponse(BaseModel):
    id: UUID
    name: str
    key: str  # показывается только при создании
    created_at: datetime


class APIKeyInfo(BaseModel):
    id: UUID
    name: str
    is_active: bool
    created_at: datetime
