"""Эмбеддинги учителя для дистилляции (relational KD в src/train.py --kd).

Учитель — самый точный состав (ViT-H+@320 + ViT-B·части, mAP@10 0.8828), он
слишком медленный для стенда, но может научить быструю модель структуре
сходств. Эмбеддинги считаются один раз на чистых (без аугментаций) кадрах
через тот же продовый экстрактор.

Для честного замера учитель должен быть holdout-моделью (не видел 308
валидационных ТС), а эмбеддинги — только для обучающей части:

    python -m src.teacher_emb --manifest runs/teacher_holdout.json --holdout \
        --out runs/teacher_holdout.npz

Файлы кладутся в runs/, а не в artifacts/: по правилам в лимит 2 ГБ идут все
.npz в каталоге решения, а эти эмбеддинги нужны только для обучения.

Для продовой дистилляции — продовый учитель и все 1541 ТС:

    python -m src.teacher_emb --manifest runs/teacher_prod.json \
        --out runs/teacher_prod.npz
"""
import argparse

import numpy as np
import pandas as pd

from .config import Config
from .dataset import split_train_val
from .extractor import Extractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--holdout", action="store_true",
                    help="только обучающая часть (без 20% валидационных ТС)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    if a.holdout:
        df, _ = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    ex = Extractor(manifest=a.manifest, graphs=False)   # графы держат лишнюю VRAM
    emb = ex.extract_frame(df, cfg.data_root / "images")
    # строки — обычным юникод-массивом: pandas 3 хранит их в своём типе, и
    # np.savez записал бы объектный массив, который не читается без pickle
    ids = np.array([str(i) for i in df.image_id], dtype="U64")
    np.savez(a.out, image_id=ids, emb=emb.astype(np.float16))
    print(f"учитель: {emb.shape} -> {a.out}")


if __name__ == "__main__":
    main()
