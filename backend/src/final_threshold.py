"""Финальный порог отказа и кривые валидации для текущего ансамбля.

Комбинация задаётся COMBO (имена наборов эмбеддингов + веса) — должна
совпадать с models/ensemble.json. Камеры-группы на валидации ИНФЕРЯТСЯ
теми же правилами, что применяются на тесте (фон 0.8 + «та же парковка»),
поэтому τ переносится честно. Пишет models/threshold.json,
models/val_curves.json и artifacts/val_report.json.
Запуск: python -m src.final_threshold
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .camera_infer import cluster, compute_descriptors
from .camera_rule_val import bbox_iou
from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_utils import compute_reid_metrics, refusal_metrics
from .postproc_experiments import aqe, dba

# (npz-файл, вес) — порядок и веса как в models/ensemble.json
# Эмбеддинги HOLDOUT-моделей того же состава, что в models/ensemble.json.
# Продовые модели видели валидационные ТС и дают на них mAP 1.0 — порог по
# ним калибровать нельзя. Веса конкатенации должны совпадать с манифестом.
COMBO = [
    ("artifacts/cand/h_320.npz", 1.0),      # DINOv3 ViT-H+ @320
    ("artifacts/cand/bp_shift.npz", 0.5),   # DINOv2 ViT-B + части-признаки @252
]
COMBO_TAG = "DINOv3 ViT-H+@320 + 0.5·ViT-B(части)@252"


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def pick(curve, tnr_floor=0.70):
    good = [c for c in curve if c["TNR"] >= tnr_floor]
    return max(good or curve, key=lambda c: c["F1"])


def main():
    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    val_q, val_g = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)

    qs, gs = [], []
    for path, w in COMBO:
        d = np.load(path)
        qs.append(w * d["q"])
        gs.append(w * d["g"])
    q, g = norm(np.concatenate(qs, 1)), norm(np.concatenate(gs, 1))
    # параметры постобработки берём из того же файла, что и инференс,
    # иначе порог калибруется не на том распределении
    pp = json.loads(Path("models/postproc.json").read_text(encoding="utf-8"))
    if pp.get("dba_k"):
        g = dba(g, k=pp["dba_k"], alpha=pp.get("alpha", 3.0))
    if pp.get("aqe_k"):
        q = aqe(q, g, k=pp["aqe_k"], alpha=pp.get("alpha", 3.0))
    sim = q @ g.T

    q_vid, g_vid = val_q["vehicle_id"].values, val_g["vehicle_id"].values
    q_cam, g_cam = val_q["camera_id"].values, val_g["camera_id"].values
    has_match = np.array([((g_vid == v) & (g_cam != c)).any()
                          for v, c in zip(q_vid, q_cam)])

    # --- камеры-группы: инференс как на тесте (фон + «та же парковка») ---
    all_df = pd.concat([val_q, val_g], ignore_index=True)
    desc = compute_descriptors(all_df, cfg)
    groups = cluster(desc, 0.80)
    qg, gg = groups[: len(val_q)], groups[len(val_q):]
    bsim = desc[: len(val_q)] @ desc[len(val_q):].T
    iou = bbox_iou(val_q[["x", "y", "w", "h"]].values.astype(float),
                   val_g[["x", "y", "w", "h"]].values.astype(float))
    same_scene = qg[:, None] == gg[None, :]
    same_scene |= (bsim >= 0.50) & (iou >= 0.85)
    same_scene |= (bsim >= 0.30) & (iou >= 0.90)

    grid = np.round(np.linspace(0.0, 1.0, 501), 4)

    def curve_of(mask):
        ms = np.where(mask, -1.0, sim).max(1)
        return ms, [refusal_metrics(ms, has_match, t) for t in grid]

    ms_inf, curve_inf = curve_of(same_scene)
    ms_true, curve_true = curve_of(q_cam[:, None] == g_cam[None, :])
    b_inf, b_true = pick(curve_inf), pick(curve_true)

    # --- метрики ранжирования (истинные камеры, протокол Market) ---
    m = compute_reid_metrics(q, g, q_vid, g_vid, q_cam, g_cam)

    print(f"ранжирование: mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} "
          f"R5={m['Rank-5']:.4f} mINP={m['mINP']:.4f}")
    print(f"инференс-камеры: τ={b_inf['threshold']:.3f} F1={b_inf['F1']:.4f} TNR={b_inf['TNR']:.4f}")
    print(f"истинные камеры: τ={b_true['threshold']:.3f} F1={b_true['F1']:.4f} TNR={b_true['TNR']:.4f}")

    Path("models/threshold.json").write_text(json.dumps({
        "threshold": b_inf["threshold"],
        "criterion": f"max F1 при TNR>=0.70; {COMBO_TAG}, DBA3/AQE1, "
                     "валидация с инференс-камерами (правило фон/IoU как на тесте)",
        "val_F1": b_inf["F1"], "val_TNR": b_inf["TNR"],
        "val_precision": b_inf["precision"], "val_recall": b_inf["recall"],
        "val_true_cameras": {k: round(float(b_true[k]), 4) for k in
                             ("F1", "TNR", "precision", "recall", "threshold")},
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    Path("models/val_curves.json").write_text(json.dumps({
        "grid": grid.tolist(),
        "F1": [c["F1"] for c in curve_inf],
        "TNR": [c["TNR"] for c in curve_inf],
        "precision": [c["precision"] for c in curve_inf],
        "recall": [c["recall"] for c in curve_inf],
        "max_sim_match": ms_inf[has_match].tolist(),
        "max_sim_nomatch": ms_inf[~has_match].tolist(),
    }), encoding="utf-8")

    Path("artifacts/val_report.json").write_text(json.dumps({
        "config": COMBO_TAG + " + DBA3/AQE1",
        "protocol": "holdout 20% идентичностей, кросс-камерный, 35% дистракторов",
        "ranking": {k: round(float(v), 4) for k, v in m.items()},
        "refusal_inferred_cams": {k: round(float(b_inf[k]), 4) for k in
                                  ("F1", "TNR", "precision", "recall", "threshold")},
        "refusal_true_cams": {k: round(float(b_true[k]), 4) for k in
                              ("F1", "TNR", "precision", "recall", "threshold")},
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved: models/threshold.json, models/val_curves.json, artifacts/val_report.json")


if __name__ == "__main__":
    main()
