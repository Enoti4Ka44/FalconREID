"""Офлайн-подбор функции уверенности для режима отказа.

Сравнивает кандидатов на «скор уверенности» по F1/TNR/AUC на валидации:
  raw    — максимальная косинусная близость (базовая линия)
  margin — отрыв топ-1 от топ-2 (s1 - s2)
  zscore — (s1 - mean(sims)) / std(sims) по запросу
  ratio  — s1 / mean(top10)
  aqe    — max sim после alpha-query-expansion запроса
  comb   — s1 + w * (s1 - s2), w подбирается по сетке

Скор должен не только максимизировать F1, но и давать высокий TNR.
"""
import json
from pathlib import Path

import numpy as np

from .eval_utils import refusal_metrics


def pr_auc(score, y):
    """AUC по кривой Precision-Recall (average precision)."""
    order = np.argsort(-score)
    y = y[order]
    cum_tp = np.cumsum(y)
    precision = cum_tp / (np.arange(len(y)) + 1)
    return float(precision[y.astype(bool)].sum() / max(y.sum(), 1))


def curve_best(score, y, tnr_floor=0.0):
    grid = np.quantile(score, np.linspace(0.001, 0.999, 400))
    rows = [refusal_metrics(score, y, t) for t in grid]
    ok = [r for r in rows if r["TNR"] >= tnr_floor] or rows
    return max(ok, key=lambda r: r["F1"])


def main():
    d = np.load("artifacts/val_embeddings.npz")
    q, g = d["q"], d["g"]
    q_vid, g_vid, q_cam, g_cam = d["q_vid"], d["g_vid"], d["q_cam"], d["g_cam"]

    sim = q @ g.T
    same_cam = q_cam[:, None] == g_cam[None, :]
    sim_x = np.where(same_cam, -np.inf, sim)          # только кросс-камерные
    has_match = np.array([((g_vid == v) & (g_cam != c)).any()
                          for v, c in zip(q_vid, q_cam)])

    S = np.sort(sim_x, axis=1)[:, ::-1]               # отсортированные близости
    s1, s2 = S[:, 0], S[:, 1]
    top10 = S[:, :10]
    finite = np.where(np.isinf(sim_x), np.nan, sim_x)
    mu = np.nanmean(finite, axis=1)
    sd = np.nanstd(finite, axis=1)

    scores = {
        "raw": s1,
        "margin": s1 - s2,
        "zscore": (s1 - mu) / (sd + 1e-9),
        "ratio": s1 / (np.abs(top10.mean(1)) + 1e-9),
    }

    # AQE: расширение запроса средним топ-3 галереи (alpha-weighted)
    k, alpha = 3, 3.0
    idx = np.argsort(-sim_x, axis=1)[:, :k]
    w = np.take_along_axis(sim_x, idx, 1) ** alpha
    q_aqe = q + (w[:, :, None] * g[idx]).sum(1)
    q_aqe /= np.linalg.norm(q_aqe, axis=1, keepdims=True)
    sim_aqe = np.where(same_cam, -np.inf, q_aqe @ g.T)
    scores["aqe"] = np.sort(sim_aqe, axis=1)[:, -1]

    # комбинация s1 и отрыва
    best_comb = None
    for w_ in [0.5, 1.0, 1.5, 2.0, 3.0]:
        sc = s1 + w_ * (s1 - s2)
        r = curve_best(sc, has_match)
        if best_comb is None or r["F1"] > best_comb[1]["F1"]:
            best_comb = (w_, r, sc)
    scores[f"comb(w={best_comb[0]})"] = best_comb[2]

    print(f"{'score':>14} | {'AUC':>6} | {'maxF1':>6} TNR@maxF1 | F1@TNR>=0.7 | F1@TNR>=0.85")
    results = {}
    for name, sc in scores.items():
        auc = pr_auc(sc, has_match)
        b0 = curve_best(sc, has_match)
        b7 = curve_best(sc, has_match, tnr_floor=0.70)
        b85 = curve_best(sc, has_match, tnr_floor=0.85)
        print(f"{name:>14} | {auc:.4f} | {b0['F1']:.4f} {b0['TNR']:>9.3f} | "
              f"{b7['F1']:.4f} (t={b7['threshold']:.3f}) | {b85['F1']:.4f}")
        results[name] = {"auc": auc, "best": b0, "tnr70": b7, "tnr85": b85}

    Path("artifacts/refusal_experiments.json").write_text(
        json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
