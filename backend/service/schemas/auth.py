"""Pydantic-схемы для аутентификации."""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


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
