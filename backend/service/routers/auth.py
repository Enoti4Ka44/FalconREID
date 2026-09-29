"""Роутер: аутентификация (регистрация, логин, API-ключи)."""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import (
    create_access_token,
    generate_api_key,
    get_current_user,
    hash_api_key,
    hash_password,
    verify_password,
)
from ..config import Settings, get_settings
from ..db import get_db
from ..models_db import APIKey, User
from ..schemas.auth import (
    APIKeyCreate,
    APIKeyInfo,
    APIKeyResponse,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201,
    summary="Регистрация пользователя",
    description="Создаёт нового пользователя. Требуется уникальное имя пользователя и пароль. "
                "После регистрации можно войти через `/api/auth/login` для получения JWT токена "
                "или создать API ключ через `/api/auth/keys`.\n\n"
                "**Тестовый пользователь:** `test` / `test` (создаётся автоматически при старте).",
)
async def register(
    body: UserRegister,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.username == body.username))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Username already exists")

    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Получение JWT токена",
    description="Аутентификация по логину и паролю. Возвращает JWT access_token, "
                "который нужно использовать в заголовке `Authorization: Bearer <token>` "
                "для доступа к защищённым эндпоинтам. Токен действителен 60 минут.\n\n"
                "**Тестовый пользователь:**\n"
                "- Логин: `test`\n"
                "- Пароль: `test`\n\n"
                "После получения токена нажмите кнопку **Authorize** вверху и введите "
                "`Bearer <полученный_токен>`.",
)
async def login(
    body: UserLogin,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(str(user.id), settings)
    return TokenResponse(access_token=token)


@router.post(
    "/keys",
    response_model=APIKeyResponse,
    status_code=201,
    summary="Создание API ключа",
    description="Создаёт новый API ключ для machine-to-machine интеграций. "
                "Ключ показывается **только один раз** при создании — сохраните его. "
                "Используйте ключ в заголовке `X-API-Key: falcon_...` для доступа к API. "
                "Для доступа требуется JWT токен.",
)
async def create_key(
    body: APIKeyCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    raw_key = generate_api_key()
    key_record = APIKey(
        user_id=user.id,
        key_hash=hash_api_key(raw_key),
        name=body.name,
    )
    db.add(key_record)
    await db.commit()
    await db.refresh(key_record)
    return APIKeyResponse(
        id=key_record.id,
        name=key_record.name,
        key=raw_key,
        created_at=key_record.created_at,
    )


@router.get(
    "/keys",
    response_model=list[APIKeyInfo],
    summary="Список API ключей",
    description="Возвращает список всех API ключей текущего пользователя. "
                "Сами ключи не возвращаются (только метаданные). "
                "Для доступа требуется JWT токен.",
)
async def list_keys(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey).where(APIKey.user_id == user.id).order_by(APIKey.created_at.desc())
    )
    keys = result.scalars().all()
    return [
        APIKeyInfo(id=k.id, name=k.name, is_active=k.is_active, created_at=k.created_at)
        for k in keys
    ]


@router.delete(
    "/keys/{key_id}",
    status_code=204,
    summary="Отзыв API ключа",
    description="Деактивирует API ключ. После отзыва ключ нельзя использовать для доступа к API. "
                "Операция необратима. Для доступа требуется JWT токен.",
)
async def revoke_key(
    key_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == user.id)
    )
    key = result.scalar_one_or_none()
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    key.is_active = False
    await db.commit()
