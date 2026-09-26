"""Полная оценка на отложенной валидации: метрики, подбор порога, анализ ошибок.

Выход:
  models/threshold.json      — обоснованный порог режима отказа
  artifacts/val_report.json  — все метрики
  artifacts/val_curves.json  — данные для графиков (PR, F1/TNR vs порог, гистограммы)
  artifacts/error_cases.json — характерные ошибки (для анализа и презентации)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dataset import split_train_val, build_val_protocol
from .eval_utils import compute_reid_metrics, refusal_metrics
from .infer import load_model
from .train import extract_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    a = ap.parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, cfg, device)

    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    val_q, val_g = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    q_emb = extract_embeddings(model, val_q, cfg, device)
    g_emb = extract_embeddings(model, val_g, cfg, device)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    # эмбеддинги валидации — для офлайн-экспериментов с постобработкой
    np.savez(a.out_dir / "val_embeddings.npz", q=q_emb, g=g_emb,
             q_vid=val_q["vehicle_id"].values, g_vid=val_g["vehicle_id"].values,
             q_cam=val_q["camera_id"].values, g_cam=val_g["camera_id"].values)

    q_ids, g_ids = val_q["vehicle_id"].values, val_g["vehicle_id"].values
    q_cams, g_cams = val_q["camera_id"].values, val_g["camera_id"].values

    # -------- ранжирование --------
    metrics = compute_reid_metrics(q_emb, g_emb, q_ids, g_ids, q_cams, g_cams)

    # -------- режим отказа --------
    sim = q_emb @ g_emb.T
    same_cam = q_cams[:, None] == g_cams[None, :]
    sim_x = np.where(same_cam, -1.0, sim)          # только кросс-камерные кандидаты
    max_sim = sim_x.max(axis=1)
    argmax = sim_x.argmax(axis=1)
    has_match = np.array([((g_ids == v) & (g_cams != c)).any()
                          for v, c in zip(q_ids, q_cams)])
    top1_correct = g_ids[argmax] == q_ids

    grid = np.round(np.linspace(0.0, 1.0, 501), 4)
    curve = [refusal_metrics(max_sim, has_match, t) for t in grid]
    best = max(curve, key=lambda m: m["F1"])

    # рабочая точка сервиса: максимум F1 при ограничении TNR >= 0.70.
    # Жюри оценивает и F1, и TNR; безусловный максимум F1 достигается при
    # TNR ~0.3 (70% ложных принятий на дистракторах) — продуктово неприемлемо.
    good = [c for c in curve if c["TNR"] >= 0.70]
    balanced = max(good, key=lambda m: m["F1"]) if good else best

    report = {
        "checkpoint": str(a.ckpt),
        "val_protocol": {"queries": len(val_q), "gallery": len(val_g),
                         "queries_with_match": int(has_match.sum()),
                         "distractor_queries": int((~has_match).sum())},
        "ranking": metrics,
        "refusal_bestF1": best,
        "refusal_balanced": balanced,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "val_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    Path("models").mkdir(exist_ok=True)
    (Path("models") / "threshold.json").write_text(json.dumps({
        "threshold": balanced["threshold"],
        "criterion": "max F1 при ограничении TNR>=0.70 (баланс точности и доли верных отказов)",
        "val_F1": balanced["F1"], "val_TNR": balanced["TNR"],
        "val_precision": balanced["precision"], "val_recall": balanced["recall"],
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    # -------- данные для графиков --------
    (a.out_dir / "val_curves.json").write_text(json.dumps({
        "grid": grid.tolist(),
        "F1": [c["F1"] for c in curve],
        "TNR": [c["TNR"] for c in curve],
        "precision": [c["precision"] for c in curve],
        "recall": [c["recall"] for c in curve],
        "max_sim_match": max_sim[has_match].tolist(),
        "max_sim_nomatch": max_sim[~has_match].tolist(),
    }))

    # -------- характерные ошибки --------
    errors = {"rank1_miss": [], "false_accept_distractor": [], "hard_correct": []}
    order = np.argsort(-sim_x, axis=1)
    for i in np.where(has_match & ~top1_correct)[0][:40]:
        errors["rank1_miss"].append({
            "query": str(val_q.iloc[i].image_id), "query_vid": int(q_ids[i]),
            "top1": str(val_g.iloc[order[i, 0]].image_id),
            "top1_vid": int(g_ids[order[i, 0]]), "sim": float(sim_x[i, order[i, 0]]),
            "true_rank": int(np.where(g_ids[order[i]] == q_ids[i])[0][0]) + 1,
        })
    thr = balanced["threshold"]
    for i in np.where(~has_match & (max_sim >= thr))[0][:40]:
        errors["false_accept_distractor"].append({
            "query": str(val_q.iloc[i].image_id), "query_vid": int(q_ids[i]),
            "top1": str(val_g.iloc[argmax[i]].image_id),
            "top1_vid": int(g_ids[argmax[i]]), "sim": float(max_sim[i]),
        })
    for i in np.where(has_match & top1_correct & (max_sim < np.quantile(max_sim[has_match], 0.1)))[0][:20]:
        errors["hard_correct"].append({
            "query": str(val_q.iloc[i].image_id), "top1": str(val_g.iloc[argmax[i]].image_id),
            "sim": float(max_sim[i]),
        })
    (a.out_dir / "error_cases.json").write_text(json.dumps(errors, indent=1))

    print(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
