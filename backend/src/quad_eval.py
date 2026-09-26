"""Оценка четверного ансамбля: L + B + C + DINOv3 (D) на holdout-валидации.

База — лучший тройной (L + 0.7B + 0.7C, mAP .8074). Сетка веса D поверх
базы + контрольные варианты замены участников. Везде DBA3+AQE1 и подбор
порога при TNR>=0.70 (протокол triple_eval, истинные камеры).
Запуск: python -m src.quad_eval
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_utils import compute_reid_metrics, refusal_metrics
from .infer import load_model
from .postproc_experiments import aqe, dba
from .train import extract_embeddings

D_BACKBONE = "vit_large_patch16_dinov3.lvd1689m"
D_IMG = 256


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def refusal_at(sim_masked, has_match, tnr_floor=0.70):
    grid = np.round(np.linspace(0, 1, 501), 4)
    curve = [refusal_metrics(sim_masked.max(1), has_match, t) for t in grid]
    good = [c for c in curve if c["TNR"] >= tnr_floor]
    return max(good or curve, key=lambda c: c["F1"])


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    val_q, val_g = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    args = (val_q["vehicle_id"].values, val_g["vehicle_id"].values,
            val_q["camera_id"].values, val_g["camera_id"].values)
    q_cam, g_cam = args[2], args[3]
    same_cam = q_cam[:, None] == g_cam[None, :]
    has_match = np.array([((args[1] == v) & (g_cam != c)).any()
                          for v, c in zip(args[0], q_cam)])

    d_npz = Path("artifacts/val_embeddings_dinov3.npz")
    if d_npz.exists():
        d = np.load(d_npz)
        qD, gD = d["q"], d["g"]
    else:
        cfgD = Config()
        cfgD.backbone, cfgD.img_size = D_BACKBONE, D_IMG
        mD = load_model("runs/holdout_dinov3/best.pt", cfgD, device)
        qD = extract_embeddings(mD, val_q, cfgD, device, bs=32)
        gD = extract_embeddings(mD, val_g, cfgD, device, bs=32)
        np.savez(d_npz, q=qD, g=gD)
        del mD
        torch.cuda.empty_cache()

    dL = np.load("artifacts/val_embeddings_vitl.npz")
    dB = np.load("artifacts/val_embeddings_vit.npz")
    dC = np.load("artifacts/val_embeddings_m2.npz")
    parts = {"L": (dL["q"], dL["g"]), "B": (dB["q"], dB["g"]),
             "C": (dC["q"], dC["g"]), "D": (qD, gD)}

    results = {}

    def evaluate(tag, combo):
        q = np.concatenate([w * parts[k][0] for k, w in combo], 1)
        g = np.concatenate([w * parts[k][1] for k, w in combo], 1)
        q, g = norm(q), norm(g)
        g = dba(g, k=3)
        q = aqe(q, g, k=1)
        m = compute_reid_metrics(q, g, *args)
        r = refusal_at(np.where(same_cam, -1.0, q @ g.T), has_match)
        print(f"{tag:26s}: mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} R5={m['Rank-5']:.4f} "
              f"mINP={m['mINP']:.4f} | F1={r['F1']:.3f} TNR={r['TNR']:.3f} thr={r['threshold']:.3f}")
        results[tag] = {"ranking": m, "refusal": r, "combo": dict(combo)}

    evaluate("D соло", [("D", 1.0)])
    evaluate("триплет (база)", [("L", 1.0), ("B", 0.7), ("C", 0.7)])
    for w in (0.3, 0.5, 0.7, 1.0):
        evaluate(f"квад +D{w}", [("L", 1.0), ("B", 0.7), ("C", 0.7), ("D", w)])
    evaluate("L+D 1.0", [("L", 1.0), ("D", 1.0)])
    evaluate("L+D+B0.7", [("L", 1.0), ("D", 1.0), ("B", 0.7)])
    evaluate("L+D+C0.7", [("L", 1.0), ("D", 1.0), ("C", 0.7)])
    evaluate("L+D0.7+B0.5+C0.5", [("L", 1.0), ("D", 0.7), ("B", 0.5), ("C", 0.5)])

    Path("runs/quad_eval.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved: runs/quad_eval.json")


if __name__ == "__main__":
    main()
