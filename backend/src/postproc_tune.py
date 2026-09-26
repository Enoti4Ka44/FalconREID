"""Тонкая настройка AQE+DBA: сетка комбинаций + влияние на режим отказа."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_utils import compute_reid_metrics, refusal_metrics
from .postproc_experiments import aqe, dba


def refusal_at(q, g, val_q, val_g, tnr_floor=0.70):
    sim = np.where(val_q["camera_id"].values[:, None] == val_g["camera_id"].values[None, :],
                   -1.0, q @ g.T)
    max_sim = sim.max(1)
    has_match = np.array([
        ((val_g["vehicle_id"].values == v) & (val_g["camera_id"].values != c)).any()
        for v, c in zip(val_q["vehicle_id"], val_q["camera_id"])])
    grid = np.round(np.linspace(0, 1, 501), 4)
    curve = [refusal_metrics(max_sim, has_match, t) for t in grid]
    good = [c for c in curve if c["TNR"] >= tnr_floor]
    return max(good, key=lambda c: c["F1"]) if good else max(curve, key=lambda c: c["F1"])


def main():
    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    val_q, val_g = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    d = np.load("artifacts/val_embeddings_vit.npz")
    q, g = d["q"], d["g"]
    args = (val_q["vehicle_id"].values, val_g["vehicle_id"].values,
            val_q["camera_id"].values, val_g["camera_id"].values)

    best = None
    for dk in (0, 1, 2, 3, 4):
        gg = dba(g, k=dk) if dk else g
        for ak in (0, 1, 2, 3):
            qq = aqe(q, gg, k=ak) if ak else q
            m = compute_reid_metrics(qq, gg, *args)
            r = refusal_at(qq, gg, val_q, val_g)
            line = (f"DBA{dk}+AQE{ak}: mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} "
                    f"mINP={m['mINP']:.4f} | F1={r['F1']:.3f} TNR={r['TNR']:.3f} "
                    f"thr={r['threshold']:.3f}")
            print(line)
            score = m["mAP"]
            if best is None or score > best[0]:
                best = (score, dk, ak, m, r)

    _, dk, ak, m, r = best
    print(f"\nBEST: DBA k={dk} + AQE k={ak} -> mAP {m['mAP']:.4f}")
    Path("models").mkdir(exist_ok=True)
    Path("models/postproc.json").write_text(json.dumps({
        "dba_k": dk, "aqe_k": ak, "alpha": 3.0,
        "val_mAP": m["mAP"], "val_R1": m["Rank-1"], "val_R5": m["Rank-5"],
        "val_mINP": m["mINP"],
        "refusal": r,
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
