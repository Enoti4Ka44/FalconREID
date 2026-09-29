"""Манифест ансамбля (models/ensemble.json) и извлечение эмбеддингов N моделей.

Единая точка правды для инференса, сервиса, индекса галереи и бенчмарка:
список (чекпоинт, backbone, размер входа, вес конкатенации).
"""
import gc
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .infer import load_model
from .train import extract_embeddings

MANIFEST = Path("models/ensemble.json")


def load_manifest(path: Path | None = None) -> list[dict]:
    p = Path(path) if path else MANIFEST
    spec = json.loads(p.read_text(encoding="utf-8"))
    return spec["models"]


def extract_ensemble(df, device, manifest=None, bs=None):
    """Эмбеддинги ансамбля для датафрейма: последовательная загрузка моделей
    (пиковая память GPU = одна модель), взвешенная конкатенация, L2-норма."""
    models = manifest or load_manifest()
    parts = []
    for m in models:
        cfg = Config()
        cfg.backbone = m["backbone"]
        cfg.img_size = m["img_size"]
        model = load_model(m["ckpt"], cfg, device)
        b = bs or (32 if "large" in m["backbone"] else 64)
        parts.append(m["weight"] * extract_embeddings(model, df, cfg, device, bs=b))
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    emb = np.concatenate(parts, axis=1)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


def weights_size_mb(manifest=None) -> float:
    """Размер весов, участвующих в ансамбле."""
    models = manifest or load_manifest()
    return sum(Path(m["ckpt"]).stat().st_size for m in models) / 2**20


def weights_size_total_mb(models_dir: Path = Path("models")) -> tuple[float, list]:
    """Размер ВСЕХ файлов весов в решении — именно так считает лимит жюри.

    Отдельно возвращает список файлов, чтобы сразу было видно лишние: вес,
    не входящий в манифест, всё равно занимает место в лимите 2048 МиБ.
    """
    used = {Path(m["ckpt"]).name for m in load_manifest()} if MANIFEST.exists() else set()
    rows = []
    for p in sorted(Path(models_dir).glob("*.pt")):
        rows.append((p.name, p.stat().st_size / 2**20, p.name in used))
    return sum(r[1] for r in rows), rows
