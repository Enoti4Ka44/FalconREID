"""Метрики постановщика на игрушечных примерах (src/official_metrics.py)."""
import numpy as np

from src.official_metrics import map_at_10, refusal_official


def test_map_at_10_perfect_and_junk_filtered():
    # запрос ТС 1 с камеры A; в галерее: 0 — то же ТС, та же камера (junk),
    # 1 — то же ТС, камера B (позитив), 2 — чужое ТС
    g_ids, g_cams = np.array([1, 1, 2]), np.array(["A", "B", "A"])
    res = map_at_10(np.array([[0, 1, 2]]), np.array([1]), g_ids, np.array(["A"]), g_cams, k=10)
    assert res["mAP@10"] == 1.0 and res["Rank-1"] == 1.0   # junk не занимает позицию


def test_map_at_10_second_place():
    g_ids, g_cams = np.array([2, 1]), np.array(["B", "B"])
    res = map_at_10(np.array([[0, 1]]), np.array([1]), g_ids, np.array(["A"]), g_cams)
    assert abs(res["mAP@10"] - 0.5) < 1e-9 and res["Rank-1"] == 0.0 and res["Rank-5"] == 1.0


def test_refusal_counts_wrong_answer_as_false_positive():
    g_ids, g_cams = np.array([1, 2]), np.array(["B", "B"])
    q_ids, q_cams = np.array([1, 1, 3]), np.array(["A", "A", "A"])
    top1 = np.array([0, 1, 0])          # верно, неверно, ответ на запрос без пары
    conf = np.array([0.9, 0.9, 0.1])    # третий — отказ
    r = refusal_official(top1, conf, q_ids, g_ids, q_cams, g_cams, tau=0.5)
    assert (r["TP"], r["FP"], r["FN"], r["TN"]) == (1, 1, 0, 1)
    # TNR = TN / (TN + FP), где FP включает и «ответил неверно» на запросе с парой:
    # консервативная (нижняя) оценка доли верных отказов на запросах без пары
    assert r["TNR"] == 0.5
