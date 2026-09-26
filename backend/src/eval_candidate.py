"""Честная оценка дообученной модели-кандидата на замороженном эталоне.

Считает эмбеддинги кандидата на валидационном протоколе (кэширует их) и
показывает: соло-метрику, что будет, если кандидатом ЗАМЕНИТЬ участника
квада, и что будет, если его ДОБАВИТЬ пятым.

    python -m src.eval_candidate --tag d_shift \
        --ckpt runs/holdout_dinov3_shift/best.pt \
        --backbone vit_large_patch16_dinov3.lvd1689m --img-size 256

Размер входа можно задать неквадратным: --img-size 320x416 (высота x ширина).
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
from .postproc_experiments import aqe, dba, norm
from .quad_eval import refusal_at
from .train import extract_embeddings

PROD_TAGS = {"L": "vitl336", "B": "vitparts", "C": "m2", "D": "dinov3"}
PROD_W = {"L": 1.0, "B": 0.7, "C": 0.5, "D": 1.0}
TUNED_W = {"L": 1.0, "B": 1.0, "C": 0.75, "D": 1.3}
TUNED_PP = dict(dba_k=4, aqe_k=1, alpha=1.0)
CACHE = Path("artifacts/cand")


def parse_size(s: str):
    if "x" in s.lower():
        h, w = s.lower().split("x")
        return (int(h), int(w))
    return int(s)


def protocol():
    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    vq, vg = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    args = (vq.vehicle_id.values, vg.vehicle_id.values,
            vq.camera_id.values, vg.camera_id.values)
    same = args[2][:, None] == args[3][None, :]
    has = np.array([((args[1] == v) & (args[3] != c)).any()
                    for v, c in zip(args[0], args[2])])
    return cfg, vq, vg, args, same, has


def embed_candidate(tag, ckpt, backbone, size, cfg, vq, vg, letterbox=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{tag}.npz"
    if path.exists():
        d = np.load(path)
        return d["q"], d["g"]
    c = Config()
    c.backbone, c.img_size, c.letterbox = backbone, size, letterbox
    m = load_model(ckpt, c, "cuda")
    q = extract_embeddings(m, vq, c, "cuda", bs=32)
    g = extract_embeddings(m, vg, c, "cuda", bs=32)
    np.savez(path, q=q, g=g)
    del m
    torch.cuda.empty_cache()
    return q, g


def eval_imagewise(a, cfg, cand_emb):
    """Соло-метрика кандидата на «тест-подобном» покадровом протоколе."""
    from .dataset import build_val_protocol_imagewise
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    vq_c, vg_c = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    vq_i, vg_i = build_val_protocol_imagewise(val_df, cfg.distractor_fraction, cfg.seed)
    # эмбеддинги уже посчитаны для камерного разбиения тех же снимков —
    # переиспользуем их по image_id, ничего не считая заново
    pool = {}
    for frame, emb in ((vq_c, cand_emb[0]), (vg_c, cand_emb[1])):
        for iid, vec in zip(frame.image_id.values, emb):
            pool[iid] = vec
    miss = [i for i in list(vq_i.image_id) + list(vg_i.image_id) if i not in pool]
    if miss:
        print(f"  (покадровый протокол пропущен: нет {len(miss)} эмбеддингов)")
        return {}
    q = norm(np.stack([pool[i] for i in vq_i.image_id]))
    g = norm(np.stack([pool[i] for i in vg_i.image_id]))
    g = dba(g, k=TUNED_PP["dba_k"], alpha=TUNED_PP["alpha"])
    q = aqe(q, g, k=TUNED_PP["aqe_k"], alpha=TUNED_PP["alpha"])
    # камеры НЕ передаём: организаторы просят упорядочить галерею по
    # принадлежности тому же ТС, без условия «с другой камеры»
    args = (vq_i.vehicle_id.values, vg_i.vehicle_id.values)
    m = compute_reid_metrics(q, g, *args)
    has = np.array([(args[1] == v).any() for v in args[0]])
    r = refusal_at(q @ g.T, has)
    print(f"\n--- покадровый протокол (как тест) ---")
    print(f"  соло {a.tag:28s} mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} "
          f"F1={r['F1']:.3f} τ={r['threshold']:.3f}")
    return {"imagewise_solo": {"mAP": round(m["mAP"], 4), "R1": round(m["Rank-1"], 4),
                               "F1": round(r["F1"], 4), "tau": round(r["threshold"], 3)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--img-size", default="256")
    ap.add_argument("--letterbox", action="store_true")
    ap.add_argument("--out", default="runs/candidates.json")
    a = ap.parse_args()

    cfg, vq, vg, args, same, has = protocol()
    P = {}
    for k, t in PROD_TAGS.items():
        d = np.load(f"artifacts/val_embeddings_{t}.npz")
        P[k] = (d["q"], d["g"])
    P["X"] = embed_candidate(a.tag, a.ckpt, a.backbone, parse_size(a.img_size),
                             cfg, vq, vg, a.letterbox)
    print(f"{a.tag}: эмбеддинг {P['X'][0].shape}")

    res = {}

    def ev(name, combo, pp=TUNED_PP):
        q = norm(np.concatenate([w * P[k][0] for k, w in combo], 1))
        g = norm(np.concatenate([w * P[k][1] for k, w in combo], 1))
        g = dba(g, k=pp["dba_k"], alpha=pp["alpha"])
        q = aqe(q, g, k=pp["aqe_k"], alpha=pp["alpha"])
        m = compute_reid_metrics(q, g, *args)
        r = refusal_at(np.where(same, -1.0, q @ g.T), has)
        res[name] = {"mAP": round(m["mAP"], 4), "R1": round(m["Rank-1"], 4),
                     "mINP": round(m["mINP"], 4), "F1": round(r["F1"], 4),
                     "TNR": round(r["TNR"], 4), "tau": round(r["threshold"], 3)}
        print(f"  {name:34s} mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} "
              f"mINP={m['mINP']:.4f} F1={r['F1']:.3f}")
        return m["mAP"]

    print("\n--- опорные точки ---")
    ev("ПРОД квад", list(PROD_W.items()), dict(dba_k=3, aqe_k=1, alpha=3.0))
    base = ev("квад ретюненный", list(TUNED_W.items()))
    print("\n--- кандидат соло ---")
    ev(f"соло {a.tag}", [("X", 1.0)])
    print("\n--- заменить участника квада ---")
    for k in ("L", "B", "C", "D"):
        for w in (0.7, 1.0, 1.3):
            combo = [(kk, TUNED_W[kk]) for kk in TUNED_W if kk != k] + [("X", w)]
            ev(f"{k} -> {a.tag} w={w}", combo)
    print("\n--- добавить пятым ---")
    for w in (0.5, 0.7, 1.0, 1.3):
        ev(f"квад + {a.tag} w={w}", list(TUNED_W.items()) + [("X", w)])

    # второй протокол: снимки делятся покадрово, как в тестовой выборке
    # организаторов (там кадры одного ТС с одной камеры попадают и в запросы,
    # и в галерею). Нужен, чтобы порог отказа калибровался на распределении,
    # похожем на реальное. См. дефект Д4 в docs/PROGRESS.md.
    res.update(eval_imagewise(a, cfg, P["X"]))

    best = max((v["mAP"], k) for k, v in res.items() if k not in
               ("ПРОД квад", "квад ретюненный"))
    print(f"\nлучшее с кандидатом: {best[1]} mAP={best[0]:.4f} "
          f"(ретюненный квад {base:.4f}, Δ={best[0]-base:+.4f})")

    out = Path(a.out)
    all_res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    all_res[a.tag] = {"ckpt": a.ckpt, "backbone": a.backbone,
                      "img_size": a.img_size, "results": res,
                      "best": {"name": best[1], "mAP": best[0],
                               "delta_vs_tuned": round(best[0] - base, 4)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
