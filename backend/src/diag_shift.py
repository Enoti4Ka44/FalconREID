"""Диагностика сдвига шкалы близостей: holdout- vs full-модели на тесте."""
import numpy as np
import pandas as pd
import torch

from .config import Config
from .infer import load_model
from .train import extract_embeddings
from .postproc_experiments import aqe, dba


def test_maxsim(ckpt1, ckpt2):
    cfg = Config()
    device = "cuda"
    q_df = pd.read_csv(cfg.data_root / "test_query.csv")
    g_df = pd.read_csv(cfg.data_root / "test_gallery.csv")
    m1 = load_model(ckpt1, cfg, device)
    q1 = extract_embeddings(m1, q_df, cfg, device)
    g1 = extract_embeddings(m1, g_df, cfg, device)
    del m1
    torch.cuda.empty_cache()
    cfg2 = Config()
    cfg2.backbone, cfg2.img_size = "convnext_base.fb_in22k_ft_in1k", 224
    m2 = load_model(ckpt2, cfg2, device)
    q2 = extract_embeddings(m2, q_df, cfg2, device)
    g2 = extract_embeddings(m2, g_df, cfg2, device)
    del m2
    torch.cuda.empty_cache()
    q = np.concatenate([q1, 0.5 * q2], 1)
    g = np.concatenate([g1, 0.5 * g2], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    g = dba(g, k=3)
    q = aqe(q, g, k=1)
    return (q @ g.T).max(1)


def stats(x, tag):
    qs = np.percentile(x, [5, 25, 50, 75, 95])
    print(f"{tag}: mean={x.mean():.3f} p5={qs[0]:.3f} p25={qs[1]:.3f} "
          f"p50={qs[2]:.3f} p75={qs[3]:.3f} p95={qs[4]:.3f}")
    return x


if __name__ == "__main__":
    h = stats(test_maxsim("runs/holdout/best.pt", "runs/holdout_convnext/best.pt"),
              "holdout-модели на ТЕСТЕ")
    f = stats(test_maxsim("models/final_vit.pt", "models/final_convnext.pt"),
              "full-модели на ТЕСТЕ    ")
    np.savez("artifacts/diag_shift.npz", holdout=h, full=f)
    # доля ниже порога 0.334
    print(f"refuse@0.334: holdout {np.mean(h < 0.334):.1%}, full {np.mean(f < 0.334):.1%}")
