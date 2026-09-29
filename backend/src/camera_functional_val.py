"""Функциональная проверка камер-групп: режим отказа на валидации,
если вместо истинных камер использовать инференс по фону кадра.

Сравнивает F1/TNR при исключении одно-групповых кандидатов против
эталона (истинные camera_id).
"""
import numpy as np
import pandas as pd

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_utils import refusal_metrics
from .postproc_experiments import aqe, dba
from .camera_infer import cluster, compute_descriptors


def refusal_curve(max_sim, has_match, tnr_floor=0.70):
    grid = np.round(np.linspace(0, 1, 501), 4)
    curve = [refusal_metrics(max_sim, has_match, t) for t in grid]
    good = [c for c in curve if c["TNR"] >= tnr_floor]
    return max(good, key=lambda c: c["F1"]) if good else max(curve, key=lambda c: c["F1"])


def main():
    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    val_q, val_g = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)

    d1 = np.load("artifacts/val_embeddings_vit.npz")
    d2 = np.load("artifacts/val_embeddings_m2.npz")
    q = np.concatenate([d1["q"], 0.5 * d2["q"]], 1)
    g = np.concatenate([d1["g"], 0.5 * d2["g"]], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    g = dba(g, k=3)
    q = aqe(q, g, k=1)
    sim = q @ g.T

    q_vid, g_vid = val_q["vehicle_id"].values, val_g["vehicle_id"].values
    q_cam, g_cam = val_q["camera_id"].values, val_g["camera_id"].values
    # истина: есть ли кросс-камерное совпадение
    has_match = np.array([((g_vid == v) & (g_cam != c)).any()
                          for v, c in zip(q_vid, q_cam)])

    # --- эталон: истинные камеры ---
    sim_true = np.where(q_cam[:, None] == g_cam[None, :], -1.0, sim)
    r_true = refusal_curve(sim_true.max(1), has_match)
    print(f"истинные камеры:      F1={r_true['F1']:.3f} TNR={r_true['TNR']:.3f} "
          f"thr={r_true['threshold']:.3f}")

    # --- инференс камер по фону ---
    all_df = pd.concat([val_q, val_g], ignore_index=True)
    desc = compute_descriptors(all_df, cfg)
    groups = cluster(desc, 0.80)
    qg, gg = groups[: len(val_q)], groups[len(val_q):]
    sim_inf = np.where(qg[:, None] == gg[None, :], -1.0, sim)
    r_inf = refusal_curve(sim_inf.max(1), has_match)
    print(f"камеры по фону (0.80): F1={r_inf['F1']:.3f} TNR={r_inf['TNR']:.3f} "
          f"thr={r_inf['threshold']:.3f}")

    # --- расширенное правило: фон ИЛИ (фон 0.65 + IoU 0.7 «та же парковка») ---
    from .camera_rule_val import bbox_iou
    bq = val_q[["x", "y", "w", "h"]].values.astype(float)
    bg_ = val_g[["x", "y", "w", "h"]].values.astype(float)
    dq, dg = desc[: len(val_q)], desc[len(val_q):]
    bsim_qg = dq @ dg.T
    iou_qg = bbox_iou(bq, bg_)
    same_ext = (qg[:, None] == gg[None, :]) | ((bsim_qg >= 0.65) & (iou_qg >= 0.70))
    sim_ext = np.where(same_ext, -1.0, sim)
    r_ext = refusal_curve(sim_ext.max(1), has_match)
    print(f"фон | (bg.65 & IoU.7): F1={r_ext['F1']:.3f} TNR={r_ext['TNR']:.3f} "
          f"thr={r_ext['threshold']:.3f}")

    # --- вообще без исключения (наивно) ---
    r_naive = refusal_curve(sim.max(1), has_match)
    print(f"без исключения:       F1={r_naive['F1']:.3f} TNR={r_naive['TNR']:.3f} "
          f"thr={r_naive['threshold']:.3f}")

    import json
    from pathlib import Path
    Path("artifacts/camera_functional_val.json").write_text(json.dumps(
        {"true_cams": r_true, "inferred": r_inf, "naive": r_naive}, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
