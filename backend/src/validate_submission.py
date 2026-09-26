"""Самопроверка артефактов сдачи: формат, порядок, согласованность.

Гоняется перед сдачей: ловит перепутанный порядок embeddings.npy,
дубликаты кандидатов, посторонние id и расхождение submission/candidates.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("E:/Задание/data"))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    a = ap.parse_args()
    q = pd.read_csv(a.data_root / "test_query.csv")
    g = pd.read_csv(a.data_root / "test_gallery.csv")
    gallery_ids = set(g.image_id)
    problems = []

    # -------- submission.csv --------
    sub = list(csv.reader(open(a.artifacts / "submission.csv")))
    header, rows = sub[0], sub[1:]
    assert header[0] == "query_id" and len(header) == 11, "заголовок submission.csv"
    if len(rows) != len(q):
        problems.append(f"submission: {len(rows)} строк, ожидалось {len(q)}")
    if [r[0] for r in rows] != list(q.image_id):
        problems.append("submission: порядок query_id не совпадает с test_query.csv")
    for r in rows:
        cands = r[1:]
        if len(set(cands)) != 10:
            problems.append(f"submission {r[0]}: дубликаты кандидатов")
            break
        if any(c not in gallery_ids for c in cands):
            problems.append(f"submission {r[0]}: неизвестный gallery_id")
            break

    # -------- embeddings.npy --------
    emb = np.load(a.artifacts / "embeddings.npy")
    n_expected = len(q) + len(g)
    if emb.shape[0] != n_expected:
        problems.append(f"embeddings: {emb.shape[0]} строк, ожидалось {n_expected}")
    if emb.dtype != np.float32:
        problems.append(f"embeddings: dtype {emb.dtype}, ожидался float32")
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        problems.append(f"embeddings: векторы не L2-нормированы (нормы {norms.min():.3f}..{norms.max():.3f})")

    # Согласованность: submission обязан быть ТОЧНЫМ косинусным ранжированием
    # embeddings.npy — это ключевое правило задачи, поэтому проверяется весь
    # топ-10, а не только первая позиция. Совпадающие с точностью до 1e-6
    # близости считаются взаимозаменяемыми (порядок между ними произволен).
    qe, ge = emb[: len(q)], emb[len(q):]
    sim = qe @ ge.T
    gid = g.image_id.values
    pos = {v: i for i, v in enumerate(gid)}
    bad_top1 = bad_order = 0
    for qi, r in enumerate(rows):
        cands = r[1:]
        if any(c not in pos for c in cands):
            continue                                   # уже отмечено выше
        scores = np.array([sim[qi, pos[c]] for c in cands])
        if np.any(np.diff(scores) > 1e-6):
            bad_order += 1                             # порядок внутри топ-10 нарушен
        kth = np.partition(sim[qi], -len(cands))[-len(cands)]
        if scores.min() < kth - 1e-6:
            bad_order += 1                             # в топ-10 попал не тот кандидат
        if gid[sim[qi].argmax()] != cands[0] and sim[qi].max() - scores[0] > 1e-6:
            bad_top1 += 1
    if bad_top1:
        problems.append(f"top1 из embeddings не совпадает с submission в {bad_top1} случаях")
    if bad_order:
        problems.append(f"порядок топ-10 не соответствует косинусному ранжированию "
                        f"в {bad_order} случаях")

    # -------- candidates.csv --------
    cand = pd.read_csv(a.artifacts / "candidates.csv", dtype=str)
    assert list(cand.columns) == ["query_id", "gallery_id", "confidence"], "заголовок candidates.csv"
    # Официальный формат (ответы постановщика 18/20/29): отказ — это ОТСУТСТВИЕ
    # строк для query_id. Строки с пустым gallery_id считаются ошибкой формата.
    empty = cand.gallery_id.isna() | (cand.gallery_id == "")
    if empty.any():
        problems.append(f"candidates: {int(empty.sum())} строк с пустым gallery_id — "
                        f"отказ должен кодироваться отсутствием строк")
    unknown_q = set(cand.query_id) - set(q.image_id)
    if unknown_q:
        problems.append(f"candidates: неизвестные query_id {list(unknown_q)[:3]}")
    accepted = cand[~empty]
    n_refused = len(set(q.image_id) - set(accepted.query_id))
    bad_ids = set(accepted.gallery_id) - gallery_ids
    if bad_ids:
        problems.append(f"candidates: неизвестные gallery_id {list(bad_ids)[:3]}")
    confs = accepted.confidence.astype(float)
    print(f"queries={len(q)} gallery={len(g)} | отказов: {n_refused} "
          f"({n_refused / len(q):.1%}), принятых пар: {len(accepted)}, "
          f"conf {confs.min():.3f}..{confs.max():.3f}" if len(accepted) else "все отказы")

    if problems:
        print("\nПРОБЛЕМЫ:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    print("OK: все артефакты корректны")


if __name__ == "__main__":
    main()
