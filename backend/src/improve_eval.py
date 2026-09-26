"""Оценка улучшений 1-4 на замороженном эталоне (sha 3fe41ad1).

Сравнивает продовый квад с вариантами, где участвует новая модель
(части-признаки / 336² / студент дистилляции). Эмбеддинги новых моделей
извлекаются один раз и кэшируются в artifacts/val_embeddings_<tag>.npz.

Запуск: python -m src.improve_eval --add vitparts=runs/holdout_vit_parts30/best.pt@252
        (несколько --add через пробел; формат tag=ckpt@img_size[@backbone])
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_utils import compute_reid_metrics
from .infer import load_model
from .postproc_experiments import aqe, dba
from .quad_eval import norm, refusal_at
from .train import extract_embeddings

# Состав ПРОДА (соответствует models/ensemble.json). Раньше здесь стояли
# наборы v1 (`vitl`, `vit`), из-за чего сравнение шло не с тем ансамблем.
BASE = {"L": "vitl336", "B": "vitparts", "D": "dinov3"}
PROD = [("L", 1.0), ("B", 0.7), ("D", 1.0)]

# ВНИМАНИЕ: замер на одном разбиении протокола ненадёжен — разброс при
# пересборке разбиения ±0.011 mAP. Для решений пользуйтесь
# `src/eval_multiseed.py` (парное сравнение по 30 сидам).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--out", default="runs/improve_eval.json")
    a = ap.parse_args()

    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    vq, vg = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    args = (vq.vehicle_id.values, vg.vehicle_id.values,
            vq.camera_id.values, vg.camera_id.values)
    same = args[2][:, None] == args[3][None, :]
    has = np.array([((args[1] == v) & (args[3] != c)).any()
                    for v, c in zip(args[0], args[2])])

    P = {}
    for k, tag in BASE.items():
        d = np.load(f"artifacts/val_embeddings_{tag}.npz")
        P[k] = (d["q"], d["g"])

    new_tags = []
    for spec in a.add:
        tag, rest = spec.split("=", 1)
        parts = rest.split("@")
        ckpt, size = parts[0], int(parts[1])
        path = Path(f"artifacts/val_embeddings_{tag}.npz")
        if not path.exists():
            c = Config()
            c.img_size = size
            if len(parts) > 2:
                c.backbone = parts[2]
            m = load_model(ckpt, c, "cuda")
            q = extract_embeddings(m, vq, c, "cuda")
            g = extract_embeddings(m, vg, c, "cuda")
            np.savez(path, q=q, g=g)
            del m
            torch.cuda.empty_cache()
        d = np.load(path)
        P[tag] = (d["q"], d["g"])
        new_tags.append(tag)
        print(f"{tag}: эмбеддинги {P[tag][0].shape}")

    results = {}

    def ev(name, combo):
        q = norm(np.concatenate([w * P[k][0] for k, w in combo], 1))
        g = norm(np.concatenate([w * P[k][1] for k, w in combo], 1))
        # та же постобработка, что в проде (models/postproc.json)
        g = dba(g, k=2, alpha=1.0)
        m = compute_reid_metrics(q, g, *args)
        r = refusal_at(np.where(same, -1.0, q @ g.T), has)
        print(f"{name:34s}: mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} R5={m['Rank-5']:.4f} "
              f"mINP={m['mINP']:.4f} | F1={r['F1']:.3f} TNR={r['TNR']:.3f} thr={r['threshold']:.3f}")
        results[name] = {"ranking": {k: round(float(v), 4) for k, v in m.items()},
                         "refusal": {k: round(float(r[k]), 4) for k in ("F1", "TNR", "threshold")},
                         "combo": dict(combo)}

    ev("ПРОД квад", PROD)
    ev("B соло", [("B", 1.0)])
    for t in new_tags:
        ev(f"{t} соло", [(t, 1.0)])
        for w in (0.7, 1.0, 1.3):
            ev(f"квад: B->{t} w{w}", [("L", 1.0), (t, w), ("C", 0.5), ("D", 1.0)])
        ev(f"квад + {t} 0.7 (5 моделей)", PROD + [(t, 0.7)])
        ev(f"квад: C->{t} 0.7", [("L", 1.0), ("B", 0.7), (t, 0.7), ("D", 1.0)])

    Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {a.out}")


if __name__ == "__main__":
    main()
