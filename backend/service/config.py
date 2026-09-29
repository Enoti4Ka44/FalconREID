"""Централизованная конфигурация сервиса."""
import json
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_root: Path = Path(os.environ.get("DATA_ROOT", "/data"))
    model_path: str = os.environ.get("MODEL_PATH", "/app/models/final_dinov3ps.pt")
    gallery_index: Path = Path(os.environ.get("GALLERY_INDEX", "/app/models/gallery_index.npz"))
    refusal_threshold: float | None = None
    val_curves: Path | None = None
    runtime_dir: Path = Path(os.environ.get("RUNTIME_DIR", "runtime"))

    # Auth (legacy)
    api_key: str | None = None

    # Database
    database_url: str = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://falcon:falcon@localhost:5432/falcon",
    )

    # JWT
    jwt_secret: str = os.environ.get("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Seed user (создаётся при старте если не существует)
    seed_username: str = os.environ.get("SEED_USERNAME", "test")
    seed_password: str = os.environ.get("SEED_PASSWORD", "test")

    # пустая переменная (REFUSAL_THRESHOLD= в docker compose) = «не задана»:
    # тогда порог берётся из models/threshold.json
    model_config = {"env_prefix": "", "case_sensitive": False, "env_ignore_empty": True}

    @property
    def threshold(self) -> float:
        if self.refusal_threshold is not None:
            return self.refusal_threshold
        thr_file = self.gallery_index.parent / "threshold.json"
        if thr_file.exists():
            try:
                return float(json.loads(thr_file.read_text(encoding="utf-8"))["threshold"])
            except Exception:
                pass
        return 0.455

    @property
    def val_curves_path(self) -> Path:
        if self.val_curves is not None:
            return self.val_curves
        return self.gallery_index.parent / "val_curves.json"

    @property
    def images_dir(self) -> Path:
        return self.data_root / "images"

    @property
    def crops_dir(self) -> Path:
        return self.data_root / "crops"

    @property
    def attn_dir(self) -> Path:
        return self.data_root / "attn"


def get_settings() -> Settings:
    return Settings()
