"""Метрики Re-ID: кросс-камерный mAP, Rank-k, mINP + метрики режима отказа."""
import numpy as np


def compute_reid_metrics(q_emb, g_emb, q_ids, g_ids, q_cams=None, g_cams=None, topk=(1, 5)):
    """mAP / Rank-k / mINP по протоколу задачи.

    Совпадения с той же камеры, что у запроса, исключаются из расчёта
    (кросс-камерный протокол — суть ReID). Запросы без единого валидного
    позитива в галерее (open-set дистракторы) в mAP не участвуют.
    """
    sim = q_emb @ g_emb.T                                   # косинусная близость (векторы L2-нормированы)
    order = np.argsort(-sim, axis=1)

    aps, inps, hits = [], [], {k: [] for k in topk}
    for i in range(len(q_ids)):
        ranked = order[i]
        good = (g_ids[ranked] == q_ids[i])
        if q_cams is not None:
            # исключаем позитивы с той же камеры полностью (стандарт Market-1501)
            same_cam_pos = (g_ids[ranked] == q_ids[i]) & (g_cams[ranked] == q_cams[i])
            ranked = ranked[~same_cam_pos]
            good = (g_ids[ranked] == q_ids[i])
        if not good.any():
            continue                                        # дистрактор — не участвует в mAP
        cum = np.cumsum(good)
        precision = cum[good] / (np.flatnonzero(good) + 1)
        aps.append(precision.mean())
        last = np.flatnonzero(good)[-1]                     # mINP: точность на позиции худшего позитива
        inps.append(cum[last] / (last + 1))
        for k in topk:
            hits[k].append(good[:k].any())

    return {
        "mAP": float(np.mean(aps)),
        "mINP": float(np.mean(inps)),
        **{f"Rank-{k}": float(np.mean(hits[k])) for k in topk},
        "num_valid_queries": len(aps),
    }


def refusal_metrics(max_sim, has_match, threshold):
    """F1 / TNR режима отказа при данном пороге.

    accept = max_sim >= threshold. Позитив = «в галерее есть совпадение».
    """
    accept = max_sim >= threshold
    tp = np.sum(accept & has_match)
    fp = np.sum(accept & ~has_match)
    fn = np.sum(~accept & has_match)
    tn = np.sum(~accept & ~has_match)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    tnr = tn / max(tn + fp, 1)
    return {"F1": float(f1), "TNR": float(tnr), "precision": float(prec),
            "recall": float(rec), "threshold": float(threshold)}


def best_refusal_threshold(max_sim, has_match, grid=None):
    """Порог, максимизирующий F1 на валидации (обоснование для защиты)."""
    if grid is None:
        grid = np.linspace(max_sim.min(), max_sim.max(), 400)
    best = max((refusal_metrics(max_sim, has_match, t) for t in grid),
               key=lambda m: m["F1"])
    return best
