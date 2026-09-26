"""Метрики в точности по ответам постановщика (Q&A «Фалькон Тех»).

До появления официального `evaluate.py` у нас это единственный источник
истины. Отличия от прежней локальной оценки (`eval_utils.py`):

1. Основная метрика — **mAP@10** по ранжированию из `submission.csv`,
   а не полный mAP по эмбеддингам. AP запроса нормируется на
   min(n_pos, 10), где n_pos — число валидных позитивов после junk-фильтра.
2. Junk-фильтр: из ранжирования запроса удаляются объекты галереи, у которых
   ОДНОВРЕМЕННО тот же vehicle_id и тот же camera_id. Объекты той же камеры
   с другим ТС остаются (это трудные негативы). Фильтр применяется ДО
   усечения до 10. Запросы без валидных позитивов исключаются из mAP.
3. Режим отказа оценивается НА УРОВНЕ ЗАПРОСА, по верхнему кандидату:
   TP — пара есть, модель ответила, верхний кандидат верный;
   FP — модель ответила, но верхний неверный, ЛИБО пары нет вообще;
   FN — пара есть, модель отказалась;
   TN — пары нет, модель отказалась.
   Прежний код засчитывал «ответил, но неверно» как TP — это завышало F1.
"""
import numpy as np


def map_at_10(ranked_gallery_idx, q_ids, g_ids, q_cams, g_cams, k=10):
    """mAP@10 и Rank-1/5 по готовому ранжированию (индексы галереи на запрос).

    ranked_gallery_idx: (n_query, >=k) — порядок галереи ПОСЛЕ всей логики
    пайплайна (то, что пишется в submission.csv). Должен быть длиннее k, если
    среди верхних позиций могут оказаться junk-объекты: они выбрасываются и
    не занимают слоты.
    """
    aps, r1, r5 = [], [], []
    for i in range(len(q_ids)):
        order = np.asarray(ranked_gallery_idx[i])
        junk = (g_ids[order] == q_ids[i]) & (g_cams[order] == q_cams[i])
        order = order[~junk]
        valid_pos = (g_ids == q_ids[i]) & (g_cams != q_cams[i])
        n_pos = int(valid_pos.sum())
        if n_pos == 0:
            continue                                  # оценивается только в отказе
        good = (g_ids[order[:k]] == q_ids[i])
        hits = np.cumsum(good)
        prec = hits / (np.arange(len(good)) + 1)
        aps.append(float((prec * good).sum() / min(n_pos, k)))
        r1.append(bool(good[:1].any()))
        r5.append(bool(good[:5].any()))
    return {"mAP@10": float(np.mean(aps)), "Rank-1": float(np.mean(r1)),
            "Rank-5": float(np.mean(r5)), "n_queries": len(aps)}


def refusal_official(top1_idx, top1_conf, q_ids, g_ids, q_cams, g_cams, tau):
    """F1 и TNR по определению постановщика, при пороге tau на confidence."""
    tp = fp = fn = tn = 0
    for i in range(len(q_ids)):
        has_pair = bool(((g_ids == q_ids[i]) & (g_cams != q_cams[i])).any())
        answered = top1_conf[i] >= tau
        if answered:
            correct = g_ids[top1_idx[i]] == q_ids[i]
            if has_pair and correct:
                tp += 1
            else:
                fp += 1
        else:
            if has_pair:
                fn += 1
            else:
                tn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    tnr = tn / max(tn + fp, 1)
    return {"F1": f1, "TNR": tnr, "precision": prec, "recall": rec,
            "tau": float(tau), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def best_refusal(top1_idx, top1_conf, q_ids, g_ids, q_cams, g_cams,
                 tnr_floor=0.70, grid=None):
    """Порог, максимизирующий официальный F1 при TNR >= tnr_floor."""
    grid = np.round(np.linspace(0, 1, 401), 4) if grid is None else grid
    curve = [refusal_official(top1_idx, top1_conf, q_ids, g_ids, q_cams, g_cams, t)
             for t in grid]
    ok = [c for c in curve if c["TNR"] >= tnr_floor]
    return max(ok or curve, key=lambda c: c["F1"]), curve
