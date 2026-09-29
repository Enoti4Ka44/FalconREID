"""Насколько по-разному модели ошибаются.

Замеры показали, что в ансамбль стоит брать не самого сильного участника, а
самого НЕПОХОЖЕГО на уже имеющихся. Этот модуль считает для каждой пары
моделей корреляцию по-запросных AP и долю совпадающих ошибок, а также
«оракула» — потолок, которого достиг бы идеальный выбор между моделями.

    python -m src.diversity
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .postproc_experiments import norm

CACHE = Path("artifacts/frozen")
PROD = {"L (ViT-L@336)": "vitl336", "B (ViT-B·части)": "vitparts",
        "C (ConvNeXt)": "m2", "D (DINOv3-L)": "dinov3"}


def per_query(q, g, qid, gid, qc, gc):
    """AP и «верен ли top-1» для каждого запроса."""
    sim = q @ g.T
    order = np.argsort(-sim, 1)
    ap = np.full(len(qid), np.nan)
    hit = np.zeros(len(qid), bool)
    for i in range(len(qid)):
        r = order[i]
        r = r[~((gid[r] == qid[i]) & (gc[r] == qc[i]))]
        good = gid[r] == qid[i]
        if not good.any():
            continue
        cum = np.cumsum(good)
        ap[i] = (cum[good] / (np.flatnonzero(good) + 1)).mean()
        hit[i] = good[0]
    return ap, hit


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default="runs/diversity.json")
    a = ap_.parse_args()

    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    vq, vg = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    qid, gid = vq.vehicle_id.values, vg.vehicle_id.values
    qc, gc = vq.camera_id.values, vg.camera_id.values

    sources = {}
    for name, tag in PROD.items():
        d = np.load(f"artifacts/val_embeddings_{tag}.npz")
        sources[name] = (d["q"], d["g"])
    for p in sorted(CACHE.glob("head_*.npz")):
        d = np.load(p)
        sources[p.stem[len("head_"):]] = (d["q"], d["g"])

    APS, HITS = {}, {}
    for name, (q, g) in sources.items():
        APS[name], HITS[name] = per_query(norm(q), norm(g), qid, gid, qc, gc)
    names = list(sources)
    valid = ~np.isnan(APS[names[0]])
    print(f"валидных запросов: {valid.sum()}\n")

    print("=== корреляция по-запросных AP (чем НИЖЕ, тем полезнее в ансамбле) ===")
    print(f"{'':22s}" + "".join(f"{n[:9]:>10s}" for n in names))
    corr = {}
    for a_ in names:
        row = []
        for b_ in names:
            c = float(np.corrcoef(APS[a_][valid], APS[b_][valid])[0, 1])
            row.append(c)
            corr[f"{a_}|{b_}"] = round(c, 3)
        print(f"{a_[:22]:22s}" + "".join(f"{v:10.2f}" for v in row))

    print("\n=== в паре с продовым квадом ===")
    prod = list(PROD)
    quad_hit = np.zeros(len(qid), bool)
    for n in prod:
        quad_hit |= HITS[n]
    print(f"{'модель':22s} {'соло mAP':>9s} {'соло R1':>8s} "
          f"{'ср.корр. с квадом':>18s} {'чинит провалов квада':>21s}")
    stats = {}
    for n in names:
        mc = float(np.mean([corr[f"{n}|{p}"] for p in prod if p != n]))
        fixes = int((HITS[n] & ~quad_hit & valid).sum())
        print(f"{n[:22]:22s} {np.nanmean(APS[n]):9.4f} {HITS[n][valid].mean():8.4f} "
              f"{mc:18.2f} {fixes:21d}")
        stats[n] = {"mAP": round(float(np.nanmean(APS[n])), 4),
                    "R1": round(float(HITS[n][valid].mean()), 4),
                    "corr_with_quad": round(mc, 3), "fixes_quad_misses": fixes}

    print(f"\nоракул по квадy (R1, если всегда выбирать правильную модель): "
          f"{quad_hit[valid].mean():.4f}")
    all_hit = quad_hit.copy()
    for n in names:
        all_hit |= HITS[n]
    print(f"оракул по всем {len(names)} моделям: {all_hit[valid].mean():.4f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"corr": corr, "stats": stats,
                                       "oracle_quad": float(quad_hit[valid].mean()),
                                       "oracle_all": float(all_hit[valid].mean())},
                                      indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
