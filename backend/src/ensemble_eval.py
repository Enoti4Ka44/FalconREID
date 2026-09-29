"""Оценка ансамбля (конкатенация эмбеддингов двух моделей) на валидации.

  python -m src.ensemble_eval --ckpt2 runs/holdout_convnext/best.pt \
      --backbone2 convnext_base.fb_in22k_ft_in1k --img-size2 224

Эмбеддинги первой модели берутся из artifacts/val_embeddings_vit.npz.
Конкатенация L2-нормированных векторов = усреднение косинусов моделей —
итоговые embeddings.npy остаются согласованными с submission.csv.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dataset import split_train_val, build_val_protocol
from .eval_utils import compute_reid_metrics, refusal_metrics
from .infer import load_model
from .train import extract_embeddings


def concat_norm(a, b, w=1.0):
    """Конкатенация с весом второй модели; L2-нормировка результата."""
    x = np.concatenate([a, w * b], axis=1)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def evaluate(q, g, val_q, val_g, tag):
    m = compute_reid_metrics(q, g, val_q["vehicle_id"].values, val_g["vehicle_id"].values,
                             val_q["camera_id"].values, val_g["camera_id"].values)
    sim = np.where(val_q["camera_id"].values[:, None] == val_g["camera_id"].values[None, :],
                   -1.0, q @ g.T)
    max_sim = sim.max(1)
    has_match = np.array([
        ((val_g["vehicle_id"].values == v) & (val_g["camera_id"].values != c)).any()
        for v, c in zip(val_q["vehicle_id"], val_q["camera_id"])])
    grid = np.round(np.linspace(0, 1, 501), 4)
    curve = [refusal_metrics(max_sim, has_match, t) for t in grid]
    good = [c for c in curve if c["TNR"] >= 0.70]
    bal = max(good, key=lambda c: c["F1"]) if good else max(curve, key=lambda c: c["F1"])
    print(f"{tag:>18}: mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} R5={m['Rank-5']:.4f} "
          f"mINP={m['mINP']:.4f} | F1={bal['F1']:.3f} TNR={bal['TNR']:.3f} "
          f"thr={bal['threshold']:.3f}")
    return {"ranking": m, "refusal": bal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt2", required=True)
    ap.add_argument("--backbone2", default="convnext_base.fb_in22k_ft_in1k")
    ap.add_argument("--img-size2", type=int, default=224)
    ap.add_argument("--weight2", type=float, default=1.0)
    a = ap.parse_args()

    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    val_q, val_g = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)

    d1 = np.load("artifacts/val_embeddings_vit.npz")
    q1, g1 = d1["q"], d1["g"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg2 = Config()
    cfg2.backbone = a.backbone2
    cfg2.img_size = a.img_size2
    model2 = load_model(a.ckpt2, cfg2, device)
    q2 = extract_embeddings(model2, val_q, cfg2, device)
    g2 = extract_embeddings(model2, val_g, cfg2, device)
    np.savez("artifacts/val_embeddings_m2.npz", q=q2, g=g2)

    from .postproc_experiments import aqe, dba

    def with_pp(q, g):
        gg = dba(g, k=3)
        return aqe(q, gg, k=1), gg

    r1 = evaluate(q1, g1, val_q, val_g, "vit only")
    evaluate(*with_pp(q1, g1), val_q, val_g, "vit+DBA3+AQE1")
    r2 = evaluate(q2, g2, val_q, val_g, "model2 only")
    results = {"vit": r1, "m2": r2}
    for w in [0.5, 0.7, 1.0]:
        qc, gc = concat_norm(q1, q2, w), concat_norm(g1, g2, w)
        results[f"ensemble_w{w}"] = evaluate(qc, gc, val_q, val_g, f"ensemble w={w}")
        results[f"ensemble_w{w}_pp"] = evaluate(*with_pp(qc, gc), val_q, val_g,
                                                f"ensemble w={w}+pp")
    (cfg.work_dir / "ensemble_eval.json").write_text(
        json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
