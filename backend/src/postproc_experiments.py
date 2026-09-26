"""Офлайн-эксперименты постобработки эмбеддингов: AQE / DBA — влияние на mAP.

Работает на artifacts/val_embeddings_vit.npz (CPU, секунды).
"""
import numpy as np
import pandas as pd

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_utils import compute_reid_metrics


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def aqe(q, g, k=3, alpha=3.0):
    """alpha-Query Expansion: запрос дополняется взвешенным средним топ-k галереи."""
    sim = q @ g.T
    idx = np.argsort(-sim, axis=1)[:, :k]
    w = np.take_along_axis(sim, idx, 1).clip(0) ** alpha
    return norm(q + (w[:, :, None] * g[idx]).sum(1))


def dba(g, k=2, alpha=3.0):
    """Database Augmentation: каждый вектор галереи дополняется соседями по галерее."""
    sim = g @ g.T
    np.fill_diagonal(sim, -1)
    idx = np.argsort(-sim, axis=1)[:, :k]
    w = np.take_along_axis(sim, idx, 1).clip(0) ** alpha
    return norm(g + (w[:, :, None] * g[idx]).sum(1))


def main():
    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    val_q, val_g = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    d = np.load("artifacts/val_embeddings_vit.npz")
    q, g = d["q"], d["g"]
    args = (val_q["vehicle_id"].values, val_g["vehicle_id"].values,
            val_q["camera_id"].values, val_g["camera_id"].values)

    def show(tag, qq, gg):
        m = compute_reid_metrics(qq, gg, *args)
        print(f"{tag:>22}: mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} "
              f"R5={m['Rank-5']:.4f} mINP={m['mINP']:.4f}")

    show("baseline", q, g)
    for k in (1, 2, 3, 5):
        show(f"AQE k={k}", aqe(q, g, k=k), g)
    for k in (1, 2, 3):
        show(f"DBA k={k}", q, dba(g, k=k))
    show("AQE2+DBA2", aqe(q, dba(g, 2), 2), dba(g, 2))


if __name__ == "__main__":
    main()
