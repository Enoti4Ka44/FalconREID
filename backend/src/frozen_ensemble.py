"""Что даёт кандидат из frozen_probe, если добавить его к продовому кваду.

Соло-метрика обманчива: в ансамбле выигрывает не самый сильный участник, а
самый некоррелированный. Здесь для каждого кандидата считается и то, и другое.

    python -m src.frozen_ensemble
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_utils import compute_reid_metrics
from .postproc_experiments import aqe, dba, norm
from .quad_eval import refusal_at

CACHE = Path("artifacts/frozen")
PROD_TAGS = {"L": "vitl336", "B": "vitparts", "C": "m2", "D": "dinov3"}
PROD_W = {"L": 1.0, "B": 0.7, "C": 0.5, "D": 1.0}
# ретюненные веса и постобработка (итерация 1)
TUNED_W = {"L": 1.0, "B": 1.0, "C": 0.75, "D": 1.3}
TUNED_PP = dict(dba_k=4, aqe_k=1, alpha=1.0)


def load_protocol():
    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    vq, vg = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    args = (vq.vehicle_id.values, vg.vehicle_id.values,
            vq.camera_id.values, vg.camera_id.values)
    same = args[2][:, None] == args[3][None, :]
    has = np.array([((args[1] == v) & (args[3] != c)).any()
                    for v, c in zip(args[0], args[2])])
    return args, same, has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/frozen_ensemble.json")
    a = ap.parse_args()

    args, same, has = load_protocol()
    P = {}
    for k, t in PROD_TAGS.items():
        d = np.load(f"artifacts/val_embeddings_{t}.npz")
        P[k] = (d["q"], d["g"])
    cands = {}
    for p in sorted(CACHE.glob("head_*.npz")):
        tag = p.stem[len("head_"):]
        d = np.load(p)
        cands[tag] = (d["q"], d["g"])
    print(f"кандидатов: {len(cands)}\n")

    results = {}

    def ev(name, combo, pp):
        q = norm(np.concatenate([w * P[k][0] for k, w in combo], 1))
        g = norm(np.concatenate([w * P[k][1] for k, w in combo], 1))
        g = dba(g, k=pp["dba_k"], alpha=pp["alpha"])
        q = aqe(q, g, k=pp["aqe_k"], alpha=pp["alpha"])
        m = compute_reid_metrics(q, g, *args)
        r = refusal_at(np.where(same, -1.0, q @ g.T), has)
        results[name] = {"mAP": round(m["mAP"], 4), "R1": round(m["Rank-1"], 4),
                         "mINP": round(m["mINP"], 4), "F1": round(r["F1"], 4),
                         "TNR": round(r["TNR"], 4), "dim": int(q.shape[1])}
        print(f"{name:44s} mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} "
              f"F1={r['F1']:.3f} dim={q.shape[1]}")
        return m["mAP"]

    base_prod = ev("ПРОД квад (веса прода, DBA3/AQE1 α3)",
                   list(PROD_W.items()), dict(dba_k=3, aqe_k=1, alpha=3.0))
    base_tuned = ev("Квад ретюненный (итерация 1)",
                    list(TUNED_W.items()), TUNED_PP)
    print()

    P.update(cands)
    for tag in cands:
        ev(f"соло {tag}", [(tag, 1.0)], TUNED_PP)
    print()
    best = {}
    for tag in cands:
        for w in (0.5, 0.7, 1.0, 1.3):
            mp = ev(f"квад+{tag} w={w}", list(TUNED_W.items()) + [(tag, w)], TUNED_PP)
            if tag not in best or mp > best[tag][0]:
                best[tag] = (mp, w)
    print("\n=== итог: прирост к ретюненному кваду ===")
    for tag, (mp, w) in sorted(best.items(), key=lambda x: -x[1][0]):
        print(f"  {tag:18s} лучший вес {w:<4} mAP={mp:.4f}  Δ={mp - base_tuned:+.4f}")
    results["_base"] = {"prod": base_prod, "tuned": base_tuned,
                        "best_add": {k: {"w": v[1], "mAP": v[0]} for k, v in best.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
