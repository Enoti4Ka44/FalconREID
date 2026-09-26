"""Порог отказа под официальную формулу балла.

Балл за режим кандидатов (ответ постановщика, вопрос 15):
    10% × (0.7 · F1 + 0.3 · TNR),
где F1/TNR считаются по запросам (micro) по ВЕРХНЕМУ кандидату:
FP — «ответил, но верхний кандидат неверный» ИЛИ «ответил, а пары нет».

В закрытом тесте 20% запросов без пары (вопрос 17), поэтому валидация
строится с долей ТС-дистракторов 0.11 — это даёт 18–20% запросов без пары.
Прежний `final_threshold.py` калибровался на ~50% дистракторов, считал
«ответил неверно» за TP и максимизировал F1 при TNR≥0.70 — все три пункта
расходятся с официальной оценкой.

Порог выбирается на сидах 42–61 и проверяется на отложенных 82–111.
Эмбеддинги — holdout-моделей того же состава, что в models/ensemble.json
(продовые модели видели валидационные ТС).

    python -m src.threshold_official
"""
import json
from pathlib import Path

import numpy as np

from .eval_multiseed import dba, load_pools
from .dataset import build_val_protocol
from .official_metrics import refusal_official
from .postproc_experiments import norm

# holdout-эмбеддинги состава из models/ensemble.json (веса совпадают)
COMBO = {"artifacts/cand/k_xbm.npz": 1.0, "artifacts/cand/bp_xbm.npz": 0.7}
DISTRACTOR_IDS = 0.11
GRID = np.round(np.arange(0.20, 0.70, 0.005), 3)


def curves(pools, val_df, seeds, keys, weights):
    pp = json.loads(Path("models/postproc.json").read_text(encoding="utf-8"))
    F1, TNR, share = [], [], []
    for s in seeds:
        vq, vg = build_val_protocol(val_df, DISTRACTOR_IDS, s)
        q = norm(np.concatenate([w * np.stack([pools[k][i] for i in vq.image_id])
                                 for k, w in zip(keys, weights)], 1))
        g = norm(np.concatenate([w * np.stack([pools[k][i] for i in vg.image_id])
                                 for k, w in zip(keys, weights)], 1))
        if pp.get("dba_k"):
            g = dba(g, pp["dba_k"], pp.get("alpha", 1.0))
        qi, gi = vq.vehicle_id.values, vg.vehicle_id.values
        qc, gc = vq.camera_id.values, vg.camera_id.values
        sim = np.where(qc[:, None] == gc[None, :], -1.0, q @ g.T)
        top, conf = sim.argmax(1), sim.max(1)
        cur = [refusal_official(top, conf, qi, gi, qc, gc, t) for t in GRID]
        F1.append([c["F1"] for c in cur])
        TNR.append([c["TNR"] for c in cur])
        has = np.array([((gi == v) & (gc != c)).any() for v, c in zip(qi, qc)])
        share.append(1 - has.mean())
    return np.array(F1), np.array(TNR), float(np.mean(share))


def main():
    extra = {f"m{i}": p for i, p in enumerate(COMBO)}
    cfg, val_df, pools = load_pools(extra)
    keys, weights = list(extra), list(COMBO.values())
    F1t, TNRt, share = curves(pools, val_df, range(42, 62), keys, weights)
    F1h, TNRh, _ = curves(pools, val_df, range(82, 112), keys, weights)
    score_t = (0.7 * F1t + 0.3 * TNRt).mean(0)
    i = int(np.argmax(score_t))
    tau = float(GRID[i])
    f1, tnr = float(F1h[:, i].mean()), float(TNRh[:, i].mean())
    res = {
        "threshold": tau,
        "criterion": "максимум 0.7·F1 + 0.3·TNR (официальная формула балла), F1/TNR "
                     "по запросам по верхнему кандидату; валидация с 20% запросов без "
                     "пары, как в закрытом тесте; отбор на сидах 42–61, проверка на 82–111",
        "composition": {k: v for k, v in COMBO.items()},
        "queries_without_pair": round(share, 3),
        "heldout_F1": round(f1, 4), "heldout_TNR": round(tnr, 4),
        "heldout_score": round(0.7 * f1 + 0.3 * tnr, 4),
        "curve_heldout": [{"tau": float(t), "F1": round(float(a), 4),
                           "TNR": round(float(b), 4)}
                          for t, a, b in zip(GRID, F1h.mean(0), TNRh.mean(0))],
    }
    Path("models/threshold.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"запросов без пары: {share:.0%}")
    print(f"τ={tau:.3f}: F1={f1:.3f} TNR={tnr:.3f} балл={0.7 * f1 + 0.3 * tnr:.4f} "
          f"(отложенные сиды)")
    print("saved: models/threshold.json")


if __name__ == "__main__":
    main()
