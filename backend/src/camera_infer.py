"""Восстановление принадлежности кадров к камерам по фону сцены.

Идея: статичный фон кадра (без зоны ТС) почти идентичен для снимков одной
камеры и различен между камерами. Дескриптор — уменьшенный кадр с
замаскированным BBox; связные компоненты по порогу похожести = камеры.

Качество проверяется на train, где истинные camera_id известны:
  python -m src.camera_infer --validate
Инференс для теста:
  python -m src.camera_infer --out artifacts/test_cameras.csv
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config, find_image

DESC_W, DESC_H = 48, 27          # характерное соотношение кадров 16:9


def frame_descriptor(img_path: Path, bbox, size=(DESC_W, DESC_H)):
    """Дескриптор фона: grayscale-миниатюра кадра с вырезанной зоной ТС."""
    im = Image.open(img_path).convert("L")
    W, H = im.size
    x, y, w, h = bbox
    arr = np.asarray(im.resize(size, Image.BILINEAR), dtype=np.float32)
    # маскируем зону ТС (в координатах миниатюры), заполняя средним фона
    sx, sy = size[0] / W, size[1] / H
    x0, y0 = int(x * sx), int(y * sy)
    x1, y1 = min(size[0], int((x + w) * sx) + 1), min(size[1], int((y + h) * sy) + 1)
    mask = np.ones_like(arr, bool)
    mask[y0:y1, x0:x1] = False
    fill = arr[mask].mean() if mask.any() else arr.mean()
    arr[~mask] = fill
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    v = arr.ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def compute_descriptors(df: pd.DataFrame, cfg: Config):
    out = np.zeros((len(df), DESC_W * DESC_H), np.float32)
    for i, r in enumerate(df.itertuples()):
        out[i] = frame_descriptor(find_image(cfg.data_root / "images", r.image_id),
                                  (r.x, r.y, r.w, r.h))
    return out


def cluster(desc: np.ndarray, thr: float):
    """Связные компоненты по cos >= thr (union-find)."""
    n = len(desc)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sim = desc @ desc.T
    for i in range(n):
        for j in np.where(sim[i, i + 1:] >= thr)[0] + i + 1:
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def validate(cfg: Config, thr: float, sample=3000, seed=42):
    df = pd.read_csv(cfg.data_root / "train.csv").sample(sample, random_state=seed)
    desc = compute_descriptors(df, cfg)
    groups = cluster(desc, thr)
    true = df["camera_id"].values
    # пары: одна камера vs один кластер
    same_cam = true[:, None] == true[None, :]
    same_grp = groups[:, None] == groups[None, :]
    iu = np.triu_indices(len(df), 1)
    sc, sg = same_cam[iu], same_grp[iu]
    prec = (sc & sg).sum() / max(sg.sum(), 1)       # кластер не склеивает камеры
    rec = (sc & sg).sum() / max(sc.sum(), 1)        # камера не дробится
    print(f"thr={thr:.2f}: кластеров={len(set(groups))} (истинных камер "
          f"{df.camera_id.nunique()}), pair-precision={prec:.4f}, pair-recall={rec:.4f}")
    return prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--thr", type=float, default=0.80)
    ap.add_argument("--out", type=Path, default=Path("artifacts/test_cameras.csv"))
    a = ap.parse_args()
    cfg = Config()

    if a.validate:
        for thr in (0.70, 0.75, 0.80, 0.85, 0.90):
            validate(cfg, thr)
        return

    q = pd.read_csv(cfg.data_root / "test_query.csv", dtype={"image_id": str}).assign(split="query")
    g = pd.read_csv(cfg.data_root / "test_gallery.csv", dtype={"image_id": str}).assign(split="gallery")
    df = pd.concat([q, g], ignore_index=True)
    desc = compute_descriptors(df, cfg)
    groups = cluster(desc, a.thr)
    df["camera_group"] = groups
    df[["image_id", "split", "camera_group"]].to_csv(a.out, index=False)
    print(f"{len(df)} кадров -> {len(set(groups))} камер-групп -> {a.out}")


if __name__ == "__main__":
    main()
