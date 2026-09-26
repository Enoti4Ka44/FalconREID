"""Центральная конфигурация пайплайна Vehicle Re-ID (Фалькон Тех, ЛЦТ-2026)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

_DATA = Path(os.environ.get("DATA_ROOT", "E:/Задание/data"))


@dataclass
class Config:
    # -------- пути --------
    data_root: Path = _DATA                            # каталог с images/, train.csv, test_*.csv
    crops_dir: Path = Path(os.environ.get("CROPS_DIR", _DATA / "crops"))  # кэш кропов ТС
    work_dir: Path = Path(os.environ.get("WORK_DIR", "runs"))  # чекпоинты и логи

    # -------- модель --------
    backbone: str = "vit_base_patch14_reg4_dinov2.lvd142m"
    img_size: int = 252            # кратно 14 для ViT/14; 336 спиллит память 12ГБ GPU
    embed_dim: int = 768           # размерность эмбеддинга (после BNNeck)
    pretrained: bool = True

    # -------- обучение --------
    epochs: int = 30
    batch_p: int = 16              # P идентичностей в батче
    batch_k: int = 4               # K снимков на идентичность
    lr: float = 1e-4               # базовый LR головы; backbone получает lr * backbone_lr_scale
    backbone_lr_scale: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    triplet_margin: float = 0.3
    triplet_weight: float = 1.0
    arcface_margin: float = 0.25   # аддитивный косинусный отступ (CosFace)
    arcface_scale: float = 48.0
    amp: bool = True
    num_workers: int = 6
    seed: int = 42

    # -------- препроцессинг --------
    bbox_pad: float = 0.06         # расширение BBox на 6% с каждой стороны
    crop_max_side: int = 640       # максимум длинной стороны при кэшировании кропа
    letterbox: bool = False        # вписывать кроп с сохранением пропорций

    # -------- валидация --------
    val_id_fraction: float = 0.2   # доля идентичностей train, отложенная под валидацию
    distractor_fraction: float = 0.35  # доля val-идентичностей, отсутствующих в галерее (open-set)

    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
