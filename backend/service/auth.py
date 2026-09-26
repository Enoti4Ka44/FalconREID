"""Аутентификация: пароли, JWT, API-ключи."""
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import Settings, get_settings
from .db import get_db
from .models_db import APIKey, User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Схемы безопасности для Swagger UI
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_access_token(user_id: str, settings: Settings) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, settings: Settings) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None


def generate_api_key() -> str:
    return f"falcon_{secrets.token_urlsafe(32)}"


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def get_current_user(
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
    credentials: str | None = Depends(bearer_scheme),
    api_key: str | None = Depends(api_key_header),
) -> User:
    """Dependency: извлекает пользователя из Bearer token или API key."""

    # 1. Проверка API key
    if api_key:
        key_hash = hash_api_key(api_key)
        result = await db.execute(
            select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active == True)
        )
        key_record = result.scalar_one_or_none()
        if key_record is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        result = await db.execute(select(User).where(User.id == key_record.user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user

    # 2. Проверка Bearer token
    if credentials:
        payload = decode_token(credentials.credentials, settings)
        if payload is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Bearer token or X-API-Key required",
    )
