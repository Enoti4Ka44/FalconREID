"""Пересчёт порога отказа из сохранённых валидационных эмбеддингов (без GPU)."""
import json
from pathlib import Path

import numpy as np

from .eval_utils import refusal_metrics


def main():
    d = np.load("artifacts/val_embeddings.npz")
    q, g = d["q"], d["g"]
    q_vid, g_vid, q_cam, g_cam = d["q_vid"], d["g_vid"], d["q_cam"], d["g_cam"]
    sim = np.where(q_cam[:, None] == g_cam[None, :], -1.0, q @ g.T)
    max_sim = sim.max(1)
    has_match = np.array([((g_vid == v) & (g_cam != c)).any()
                          for v, c in zip(q_vid, q_cam)])
    grid = np.round(np.linspace(0.0, 1.0, 501), 4)
    curve = [refusal_metrics(max_sim, has_match, t) for t in grid]
    best = max(curve, key=lambda m: m["F1"])
    good = [c for c in curve if c["TNR"] >= 0.70]
    balanced = max(good, key=lambda m: m["F1"]) if good else best
    Path("models").mkdir(exist_ok=True)
    Path("models/threshold.json").write_text(json.dumps({
        "threshold": balanced["threshold"],
        "criterion": "max F1 при ограничении TNR>=0.70 (баланс точности и доли верных отказов)",
        "val_F1": balanced["F1"], "val_TNR": balanced["TNR"],
        "val_precision": balanced["precision"], "val_recall": balanced["recall"],
        "alt_maxF1": best,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(balanced, indent=1))


if __name__ == "__main__":
    main()
