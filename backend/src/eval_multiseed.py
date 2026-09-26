"""Парное сравнение составов на НЕСКОЛЬКИХ переразбиениях протокола.

Замер на одном разбиении (сид 42) обманчив: разброс mAP между
переразбиениями тех же 308 валидационных ТС составляет ±0.011 — втрое больше
заявленного в документации ±0.003. Поэтому любое сравнение ведётся ПАРНО:
оба состава считаются на одном и том же разбиении, и усредняется разница.

Кэш эмбеддингов покрывает все снимки валидационных ТС, поэтому пересборка
протокола не требует ни одного прогона модели — только переиндексацию.

    python -m src.eval_multiseed --add cand=artifacts/cand/cand.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_utils import compute_reid_metrics, refusal_metrics
from .postproc_experiments import norm

PROD_TAGS = {"L": "vitl336", "B": "vitparts", "C": "m2", "D": "dinov3"}
PROD_W = {"L": 1.0, "B": 0.7, "C": 0.5, "D": 1.0}
PROD_PP = dict(dba_k=3, aqe_k=1, alpha=3.0)
N_SEEDS = 10


def dba(g, k, al):
    if k <= 0:
        return g
    s = g @ g.T
    np.fill_diagonal(s, -1)
    i = np.argsort(-s, 1)[:, :k]
    w = np.take_along_axis(s, i, 1).clip(0) ** al
    return norm(g + (w[:, :, None] * g[i]).sum(1))


def aqe(q, g, k, al):
    if k <= 0:
        return q
    s = q @ g.T
    i = np.argsort(-s, 1)[:, :k]
    w = np.take_along_axis(s, i, 1).clip(0) ** al
    return norm(q + (w[:, :, None] * g[i]).sum(1))


def load_pools(extra=None):
    """image_id -> вектор, для каждого участника. Строится из кэша сида 42."""
    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    vq0, vg0 = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    pools = {}
    for k, t in PROD_TAGS.items():
        d = np.load(f"artifacts/val_embeddings_{t}.npz")
        pools[k] = {**dict(zip(vq0.image_id.values, d["q"])),
                    **dict(zip(vg0.image_id.values, d["g"]))}
    for tag, path in (extra or {}).items():
        d = np.load(path)
        pools[tag] = {**dict(zip(vq0.image_id.values, d["q"])),
                      **dict(zip(vg0.image_id.values, d["g"]))}
    return cfg, val_df, pools


def evaluate(pools, val_df, cfg, combo, pp, seeds=range(42, 42 + N_SEEDS)):
    """Метрики состава на каждом переразбиении."""
    out = []
    for seed in seeds:
        vq, vg = build_val_protocol(val_df, cfg.distractor_fraction, seed)
        args = (vq.vehicle_id.values, vg.vehicle_id.values,
                vq.camera_id.values, vg.camera_id.values)
        q = norm(np.concatenate(
            [w * np.stack([pools[k][i] for i in vq.image_id]) for k, w in combo], 1))
        g = norm(np.concatenate(
            [w * np.stack([pools[k][i] for i in vg.image_id]) for k, w in combo], 1))
        g = dba(g, pp["dba_k"], pp["alpha"])
        q = aqe(q, g, pp["aqe_k"], pp["alpha"])
        m = compute_reid_metrics(q, g, *args)
        same = args[2][:, None] == args[3][None, :]
        has = np.array([((args[1] == v) & (args[3] != c)).any()
                        for v, c in zip(args[0], args[2])])
        ms = np.where(same, -1.0, q @ g.T).max(1)
        grid = np.round(np.linspace(0, 1, 501), 4)
        cur = [refusal_metrics(ms, has, t) for t in grid]
        ok = [c for c in cur if c["TNR"] >= 0.70]
        r = max(ok or cur, key=lambda c: c["F1"])
        out.append({"mAP": m["mAP"], "R1": m["Rank-1"], "mINP": m["mINP"],
                    "F1": r["F1"], "TNR": r["TNR"], "tau": r["threshold"]})
    return out


def summarize(name, runs, base=None):
    mp = np.array([r["mAP"] for r in runs])
    r1 = np.array([r["R1"] for r in runs])
    f1 = np.array([r["F1"] for r in runs])
    line = (f"{name:38s} mAP={mp.mean():.4f}±{mp.std():.4f} "
            f"R1={r1.mean():.4f} F1={f1.mean():.3f}")
    res = {"mAP": round(float(mp.mean()), 4), "mAP_std": round(float(mp.std()), 4),
           "R1": round(float(r1.mean()), 4), "F1": round(float(f1.mean()), 4)}
    if base is not None:
        d = mp - np.array([r["mAP"] for r in base])
        se = d.std(ddof=1) / np.sqrt(len(d))
        verdict = ("подтверждено" if abs(d.mean()) > 2 * se else "в пределах шума")
        line += f" | Δ={d.mean():+.4f}±{se:.4f} ({(d > 0).sum()}/{len(d)}) {verdict}"
        res.update({"delta": round(float(d.mean()), 4), "se": round(float(se), 4),
                    "wins": int((d > 0).sum()), "verdict": verdict})
    print(line, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[],
                    help="tag=путь_к_npz (эмбеддинги на разбиении сида 42)")
    ap.add_argument("--out", default="runs/multiseed.json")
    a = ap.parse_args()

    extra = dict(s.split("=", 1) for s in a.add)
    cfg, val_df, pools = load_pools(extra)
    print(f"сидов протокола: {N_SEEDS}, участников: {list(pools)}\n")

    base = evaluate(pools, val_df, cfg, list(PROD_W.items()), PROD_PP)
    res = {"ПРОД квад": summarize("ПРОД квад", base)}
    res["ретюн весов+постобработки"] = summarize(
        "ретюн весов+постобработки",
        evaluate(pools, val_df, cfg, [("L", 1.0), ("B", 1.0), ("C", 0.75), ("D", 1.3)],
                 dict(dba_k=4, aqe_k=1, alpha=1.0)), base)
    res["без ConvNeXt"] = summarize(
        "без ConvNeXt",
        evaluate(pools, val_df, cfg,
                 [("L", 1.0), ("B", 0.7), ("D", 1.0)], PROD_PP), base)
    for tag in extra:
        res[f"квад: C -> {tag}"] = summarize(
            f"квад: C -> {tag}",
            evaluate(pools, val_df, cfg,
                     [("L", 1.0), ("B", 0.7), (tag, 0.5), ("D", 1.0)], PROD_PP), base)
        res[f"квад: B -> {tag}"] = summarize(
            f"квад: B -> {tag}",
            evaluate(pools, val_df, cfg,
                     [("L", 1.0), (tag, 0.7), ("C", 0.5), ("D", 1.0)], PROD_PP), base)
        res[f"квад + {tag}"] = summarize(
            f"квад + {tag}",
            evaluate(pools, val_df, cfg,
                     list(PROD_W.items()) + [(tag, 0.7)], PROD_PP), base)
        res[f"соло {tag}"] = summarize(
            f"соло {tag}",
            evaluate(pools, val_df, cfg, [(tag, 1.0)], PROD_PP))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
