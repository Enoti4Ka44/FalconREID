"""Эксперименты с мерой уверенности режима отказа (тройной ансамбль).

Вопрос: даёт ли нормализация скора (margin/z-score) или цветовое вето
лучший F1 при TNR>=0.70, чем сырой max-косинус с глобальным порогом.
Запуск: python -m src.confidence_tune
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_utils import refusal_metrics
from .postproc_experiments import aqe, dba


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def best_at_tnr(conf, has_match, tnr_floor=0.70, n=501):
    grid = np.linspace(conf.min(), conf.max(), n)
    curve = [refusal_metrics(conf, has_match, t) for t in grid]
    good = [c for c in curve if c["TNR"] >= tnr_floor]
    return max(good or curve, key=lambda c: c["F1"])


def color_sig(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = im.crop((int(w * 0.2), int(h * 0.25), int(w * 0.8), int(h * 0.85)))
    hsv = np.asarray(im.convert("HSV"), np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chroma = (S > 60) & (V > 40)
    if chroma.mean() < 0.10:
        return np.nan                       # ахроматический кузов
    return float(np.median(H[chroma]) * 360 / 255)


def hue_dist(h1, h2):
    d = np.abs(h1 - h2)
    return np.minimum(d, 360 - d)


def main():
    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    val_q, val_g = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    q_vid, g_vid = val_q["vehicle_id"].values, val_g["vehicle_id"].values
    q_cam, g_cam = val_q["camera_id"].values, val_g["camera_id"].values
    same_cam = q_cam[:, None] == g_cam[None, :]
    has_match = np.array([((g_vid == v) & (g_cam != c)).any()
                          for v, c in zip(q_vid, q_cam)])

    qL, gL = (lambda d: (d["q"], d["g"]))(np.load("artifacts/val_embeddings_vitl.npz"))
    dB = np.load("artifacts/val_embeddings_vit.npz")
    dC = np.load("artifacts/val_embeddings_m2.npz")
    q = norm(np.concatenate([qL, 0.7 * dB["q"], 0.7 * dC["q"]], 1))
    g = norm(np.concatenate([gL, 0.7 * dB["g"], 0.7 * dC["g"]], 1))
    g = dba(g, k=3)
    q = aqe(q, g, k=1)

    sim = q @ g.T
    sim_x = np.where(same_cam, -1.0, sim)          # кросс-камерные кандидаты
    order = np.argsort(-sim_x, axis=1)
    top1 = sim_x[np.arange(len(q)), order[:, 0]]
    top2 = sim_x[np.arange(len(q)), order[:, 1]]

    results = {}

    def report(tag, conf):
        r = best_at_tnr(conf, has_match)
        print(f"{tag:34s}: F1={r['F1']:.4f} TNR={r['TNR']:.4f} "
              f"P={r['precision']:.3f} R={r['recall']:.3f} thr={r['threshold']:.4f}")
        results[tag] = r

    report("baseline max-cos", top1)
    report("margin top1-top2", top1 - top2)
    report("ratio top1/top2", top1 / np.maximum(top2, 1e-6))
    mu = sim_x.mean(1, where=sim_x > -1)
    sd = sim_x.std(1, where=sim_x > -1)
    report("z-score", (top1 - mu) / np.maximum(sd, 1e-6))
    for a in (0.3, 0.5, 0.7):
        report(f"blend cos+{a}*margin", top1 + a * (top1 - top2))

    # --- цветовое вето: разный цвет кузова понижает уверенность топ-1 ---
    print("считаю цвета кропов ...")
    hq = np.array([color_sig(cfg.crops_dir / f"{i}.jpg") for i in val_q.image_id])
    hg = np.array([color_sig(cfg.crops_dir / f"{i}.jpg") for i in val_g.image_id])
    h_top1 = hg[order[:, 0]]
    both = ~np.isnan(hq) & ~np.isnan(h_top1)
    mismatch = both & (hue_dist(hq, h_top1) > 40)
    print(f"хроматические пары: {both.sum()}/{len(q)}, цветовой мисматч топ-1: {mismatch.sum()}")
    for pen in (0.03, 0.05, 0.08, 0.12):
        report(f"color veto -{pen}", top1 - pen * mismatch)

    Path("runs/confidence_tune.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
