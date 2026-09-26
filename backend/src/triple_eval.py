"""Сравнение финальных конфигураций ансамбля на отложенной валидации.

  python -m src.triple_eval --vitl runs/holdout_vitl/best.pt

Оценивает (везде + DBA3/AQE1 и подбор порога при TNR>=0.70):
  - ViT-L соло
  - ViT-L + ConvNeXt (сетка веса w2)
  - ViT-L + ViT-B (сетка w2)
  - тройной: ViT-L + w_b*ViT-B + w_c*ConvNeXt (сетка весов)
Плюс сводка по суммарному размеру файлов весов каждой конфигурации
(лимит ТЗ — 2 ГБ) и по латентности (число прогонов backbone).
"""
import argparse
import itertools
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

GB = 2**30


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def refusal_at(sim_masked, has_match, tnr_floor=0.70):
    grid = np.round(np.linspace(0, 1, 501), 4)
    curve = [refusal_metrics(sim_masked.max(1), has_match, t) for t in grid]
    good = [c for c in curve if c["TNR"] >= tnr_floor]
    return max(good or curve, key=lambda c: c["F1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitl", default="runs/holdout_vitl/best.pt")
    a = ap.parse_args()
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

    # --- эмбеддинги трёх моделей на валидации ---
    vitl_npz = Path("artifacts/val_embeddings_vitl.npz")
    if vitl_npz.exists():
        d = np.load(vitl_npz)
        qL, gL = d["q"], d["g"]
    else:
        cfgL = Config()
        cfgL.backbone = "vit_large_patch14_reg4_dinov2.lvd142m"
        cfgL.img_size = 224
        mL = load_model(a.vitl, cfgL, device)
        qL = extract_embeddings(mL, val_q, cfgL, device, bs=32)
        gL = extract_embeddings(mL, val_g, cfgL, device, bs=32)
        np.savez(vitl_npz, q=qL, g=gL)
        del mL
        torch.cuda.empty_cache()
    dB = np.load("artifacts/val_embeddings_vit.npz")
    dC = np.load("artifacts/val_embeddings_m2.npz")
    qB, gB, qC, gC = dB["q"], dB["g"], dC["q"], dC["g"]

    sizes = {  # МБ файлов весов
        "L": Path(a.vitl).stat().st_size / GB * 1024 if Path(a.vitl).exists() else 1160,
        "B": Path("models/final_vit.pt").stat().st_size / GB * 1024,
        "C": Path("models/final_convnext.pt").stat().st_size / GB * 1024,
    }

    def evaluate(tag, parts, weights):
        q = np.concatenate([w * p[0] for w, p in zip(weights, parts)], 1)
        g = np.concatenate([w * p[1] for w, p in zip(weights, parts)], 1)
        q, g = norm(q), norm(g)
        g = dba(g, k=3)
        q = aqe(q, g, k=1)
        m = compute_reid_metrics(q, g, *args)
        r = refusal_at(np.where(same_cam, -1.0, q @ g.T), has_match)
        print(f"{tag:32s}: mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} "
              f"mINP={m['mINP']:.4f} | F1={r['F1']:.3f} TNR={r['TNR']:.3f} "
              f"thr={r['threshold']:.3f}")
        return {"ranking": m, "refusal": r, "weights": weights}

    results = {}
    results["L"] = evaluate(f"ViT-L соло ({sizes['L']:.0f}МБ)", [(qL, gL)], [1.0])
    for w in (0.3, 0.5, 0.7, 1.0):
        results[f"L+C w{w}"] = evaluate(
            f"L+ConvNeXt w={w} ({sizes['L']+sizes['C']:.0f}МБ)",
            [(qL, gL), (qC, gC)], [1.0, w])
    for w in (0.3, 0.5, 0.7, 1.0):
        results[f"L+B w{w}"] = evaluate(
            f"L+ViT-B w={w} ({sizes['L']+sizes['B']:.0f}МБ)",
            [(qL, gL), (qB, gB)], [1.0, w])
    for wb, wc in itertools.product((0.3, 0.5, 0.7), (0.3, 0.5, 0.7)):
        results[f"L+B{wb}+C{wc}"] = evaluate(
            f"тройной B={wb} C={wc} ({sum(sizes.values()):.0f}МБ)",
            [(qL, gL), (qB, gB), (qC, gC)], [1.0, wb, wc])

    print(f"\nвеса файлов: L={sizes['L']:.0f} B={sizes['B']:.0f} C={sizes['C']:.0f} МБ; "
          f"тройной={sum(sizes.values()):.0f} МБ (лимит 2048 МБ)")
    best = max(results.items(), key=lambda kv: kv[1]["ranking"]["mAP"])
    print(f"ЛУЧШИЙ по mAP: {best[0]} -> {best[1]['ranking']['mAP']:.4f}")
    Path("runs/triple_eval.json").write_text(
        json.dumps(results, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
