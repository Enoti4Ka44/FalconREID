"""Сводка абляций рецепта обучения: парное сравнение с контролем по сидам.

Все варианты обучены на одном backbone, поэтому чистый сигнал даёт
СОЛО-сравнение: у кого лучше эмбеддинг, тот и выиграет и в ансамбле.
Сравнение парное — оба варианта считаются на одном переразбиении протокола.

    python -m src.ablation_report --base b_base
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_multiseed import aqe, dba
from .eval_utils import compute_reid_metrics, refusal_metrics
from .postproc_experiments import norm

CAND = Path("artifacts/cand")
PP = dict(dba_k=2, aqe_k=0, alpha=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="b_base")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="runs/ablation_report.json")
    a = ap.parse_args()

    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    vq0, vg0 = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)

    pools = {}
    for p in sorted(CAND.glob("*.npz")):
        d = np.load(p)
        pools[p.stem] = {**dict(zip(vq0.image_id.values, d["q"])),
                         **dict(zip(vg0.image_id.values, d["g"]))}
    if a.base not in pools:
        print(f"нет контроля {a.base}; есть: {list(pools)}")
        return
    print(f"вариантов: {len(pools)}, контроль: {a.base}, сидов: {a.seeds}\n")

    seeds = list(range(42, 42 + a.seeds))
    splits = []
    for s in seeds:
        vq, vg = build_val_protocol(val_df, cfg.distractor_fraction, s)
        splits.append((vq, vg))

    def run(tag):
        maps, r1s, f1s = [], [], []
        for vq, vg in splits:
            q = norm(np.stack([pools[tag][i] for i in vq.image_id]))
            g = norm(np.stack([pools[tag][i] for i in vg.image_id]))
            g = dba(g, PP["dba_k"], PP["alpha"])
            q = aqe(q, g, PP["aqe_k"], PP["alpha"])
            args = (vq.vehicle_id.values, vg.vehicle_id.values,
                    vq.camera_id.values, vg.camera_id.values)
            m = compute_reid_metrics(q, g, *args)
            same = args[2][:, None] == args[3][None, :]
            has = np.array([((args[1] == v) & (args[3] != c)).any()
                            for v, c in zip(args[0], args[2])])
            ms = np.where(same, -1.0, q @ g.T).max(1)
            cur = [refusal_metrics(ms, has, t) for t in np.round(np.linspace(0, 1, 301), 4)]
            ok = [c for c in cur if c["TNR"] >= 0.70]
            maps.append(m["mAP"])
            r1s.append(m["Rank-1"])
            f1s.append(max(ok or cur, key=lambda c: c["F1"])["F1"])
        return np.array(maps), np.array(r1s), np.array(f1s)

    base = run(a.base)
    print(f"{a.base:12s} (контроль)  mAP={base[0].mean():.4f}±{base[0].std():.4f} "
          f"R1={base[1].mean():.4f} F1={base[2].mean():.3f}")
    rows, out = [], {}
    for tag in sorted(pools):
        if tag == a.base:
            continue
        v = run(tag)
        d = v[0] - base[0]
        se = d.std(ddof=1) / np.sqrt(len(d))
        rows.append((d.mean(), se, tag, v))
        out[tag] = {"mAP": round(float(v[0].mean()), 4),
                    "R1": round(float(v[1].mean()), 4),
                    "F1": round(float(v[2].mean()), 4),
                    "delta": round(float(d.mean()), 4), "se": round(float(se), 4),
                    "wins": int((d > 0).sum())}
    print()
    for dm, se, tag, v in sorted(rows, reverse=True):
        verdict = "ПОДТВЕРЖДЕНО" if abs(dm) > 2 * se else "шум"
        print(f"{tag:12s} mAP={v[0].mean():.4f} R1={v[1].mean():.4f} F1={v[2].mean():.3f} "
              f"| Δ={dm:+.4f}±{se:.4f} ({int((v[0] - base[0] > 0).sum())}/{len(seeds)}) "
              f"{verdict}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
