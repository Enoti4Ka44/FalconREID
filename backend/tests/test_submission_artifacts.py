"""Формат и согласованность артефактов сдачи (ТЗ §8), без GPU и без датасета.

Полная проверка против test_query.csv / test_gallery.csv — `python -m src.validate_submission`.
"""
import csv
import json

import numpy as np
import pytest

from conftest import ROOT

ART = ROOT / "artifacts"
N_QUERY, N_GALLERY = 1110, 750


@pytest.fixture(scope="module")
def submission():
    rows = list(csv.reader(open(ART / "submission.csv", encoding="utf-8")))
    return rows[0], rows[1:]


@pytest.fixture(scope="module")
def embeddings():
    return np.load(ART / "embeddings.npy")


def test_submission_header_and_shape(submission):
    header, rows = submission
    assert header == ["query_id"] + [f"gallery_id_{i}" for i in range(1, 11)]
    assert len(rows) == N_QUERY
    assert len({r[0] for r in rows}) == N_QUERY, "query_id должны быть уникальны"
    for r in rows:
        assert len(r) == 11 and len(set(r[1:])) == 10, f"{r[0]}: нужно 10 разных кандидатов"


def test_embeddings_order_shape_norm(embeddings):
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (N_QUERY + N_GALLERY, 3072)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3)


def test_submission_is_cosine_ranking(submission, embeddings):
    """Топ-10 submission — точное косинусное ранжирование embeddings.npy.

    Без test_gallery.csv порядок галереи неизвестен, поэтому он восстанавливается
    голосованием: gallery_id на позиции j у запроса i соответствует строке
    argsort(-sim[i])[j]. Затем для КАЖДОГО запроса проверяется, что близости
    его 10 кандидатов совпадают с 10 наибольшими близостями (с точностью до ничьих).
    """
    from collections import Counter
    _, rows = submission
    q, g = embeddings[:N_QUERY], embeddings[N_QUERY:]
    sim = q @ g.T
    order = np.argsort(-sim, axis=1)[:, :10]
    votes = {}
    for i, r in enumerate(rows):
        for gid, row in zip(r[1:], order[i]):
            votes.setdefault(gid, Counter())[int(row)] += 1
    id2row = {gid: c.most_common(1)[0][0] for gid, c in votes.items()}
    assert len(set(id2row.values())) == len(id2row), "разные id сопоставились одной строке"
    top10 = -np.sort(-sim, axis=1)[:, :10]
    for i, r in enumerate(rows):
        got = np.array([sim[i, id2row[gid]] for gid in r[1:]])
        assert np.allclose(got, top10[i], atol=1e-5), f"запрос {r[0]}: не косинусный топ-10"


def test_candidates_refusal_format():
    thr = json.loads((ROOT / "models/threshold.json").read_text(encoding="utf-8"))["threshold"]
    rows = list(csv.DictReader(open(ART / "candidates.csv", encoding="utf-8")))
    assert rows, "candidates.csv пуст"
    assert set(rows[0]) == {"query_id", "gallery_id", "confidence"}
    per_query = {}
    for r in rows:
        assert r["gallery_id"], "отказ кодируется отсутствием строк, а не пустым gallery_id"
        c = float(r["confidence"])
        assert thr - 1e-4 <= c <= 1.0 + 1e-4, "принятые кандидаты не ниже порога"
        per_query.setdefault(r["query_id"], []).append(c)
    for qid, confs in per_query.items():
        assert len(confs) <= 10
        assert confs == sorted(confs, reverse=True), f"{qid}: уверенность должна убывать"
    sub_q = {r[0] for r in csv.reader(open(ART / "submission.csv", encoding="utf-8"))}
    assert set(per_query) <= sub_q
