"""Проверка расширенного правила одно-сценности на train (истинные камеры)."""
import numpy as np
import pandas as pd

from .config import Config
from .camera_infer import compute_descriptors


def bbox_iou(b1, b2):
    x1, y1, w1, h1 = b1.T
    x2, y2, w2, h2 = b2.T
    xa = np.maximum(x1[:, None], x2[None, :])
    ya = np.maximum(y1[:, None], y2[None, :])
    xb = np.minimum((x1 + w1)[:, None], (x2 + w2)[None, :])
    yb = np.minimum((y1 + h1)[:, None], (y2 + h2)[None, :])
    inter = np.clip(xb - xa, 0, None) * np.clip(yb - ya, 0, None)
    union = (w1 * h1)[:, None] + (w2 * h2)[None, :] - inter
    return inter / np.maximum(union, 1)


def main(sample=2500, seed=42):
    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv").sample(sample, random_state=seed)
    desc = compute_descriptors(df, cfg)
    bsim = desc @ desc.T
    iou = bbox_iou(df[["x", "y", "w", "h"]].values.astype(float),
                   df[["x", "y", "w", "h"]].values.astype(float))
    true_same = df.camera_id.values[:, None] == df.camera_id.values[None, :]
    iu = np.triu_indices(len(df), 1)

    for thr_hi, thr_lo, iou_thr in [(0.80, None, None), (0.80, 0.65, 0.70),
                                    (0.80, 0.50, 0.85), (0.80, 0.45, 0.90),
                                    (0.80, 0.30, 0.90), (0.80, 0.00, 0.90)]:
        pred = bsim >= thr_hi
        if thr_lo is not None:
            pred |= (bsim >= thr_lo) & (iou >= iou_thr)
        p, t = pred[iu], true_same[iu]
        prec = (p & t).sum() / max(p.sum(), 1)
        rec = (p & t).sum() / max(t.sum(), 1)
        rule = f"bg>={thr_hi}" + (f" | (bg>={thr_lo} & IoU>={iou_thr})" if thr_lo is not None else "")
        print(f"{rule:38s}: precision={prec:.4f} recall={rec:.4f} pairs={p.sum()}")


if __name__ == "__main__":
    main()
