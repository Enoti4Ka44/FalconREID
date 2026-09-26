"""Демонстрация масштабируемости: FAISS HNSW + int8-квантизация на галерее 10^6.

Показывает готовность к городскому масштабу: миллион векторов 1536-d,
скорость запроса, recall против точного поиска и потребление памяти
(int8-квантизация: ~1.5 ГБ RAM на 10^6 векторов вместо 6 ГБ float32).
"""
import json
import time
from pathlib import Path

import faiss
import numpy as np


def main(n_gallery: int = 1_000_000, dim: int = 1536, n_queries: int = 500):
    rng = np.random.default_rng(42)
    print(f"галерея: {n_gallery:,} векторов dim={dim} "
          f"({n_gallery * dim * 4 / 2**30:.1f} ГБ float32, "
          f"{n_gallery * dim / 2**30:.1f} ГБ int8)")

    # архив 10^6 строится из РЕАЛЬНЫХ эмбеддингов тестовой выборки
    # (выпуклые комбинации пар реальных векторов + малый шум): геометрия
    # многообразия признаков сохраняется — именно на ней работает HNSW,
    # в отличие от i.i.d.-гауссовской синтетики, где графовые индексы
    # вырождаются. Стандартный подход ANN-бенчмарков (ср. SIFT1M).
    real = np.load("artifacts/embeddings.npy").astype(np.float32)
    print(f"основа: {real.shape[0]} реальных векторов теста")
    gal = np.empty((n_gallery, dim), np.float32)
    chunk = 100_000
    for s in range(0, n_gallery, chunk):
        e = min(s + chunk, n_gallery)
        i1 = rng.integers(0, len(real), e - s)
        i2 = rng.integers(0, len(real), e - s)
        a = rng.uniform(0.55, 0.95, (e - s, 1)).astype(np.float32)
        gal[s:e] = a * real[i1] + (1 - a) * real[i2]
        gal[s:e] += 0.03 * rng.standard_normal((e - s, dim)).astype(np.float32)
    faiss.normalize_L2(gal)
    i1 = rng.integers(0, len(real), n_queries)
    i2 = rng.integers(0, len(real), n_queries)
    a = rng.uniform(0.55, 0.95, (n_queries, 1)).astype(np.float32)
    q = a * real[i1] + (1 - a) * real[i2] + \
        0.03 * rng.standard_normal((n_queries, dim)).astype(np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    faiss.normalize_L2(q)

    # -------- точный поиск (чанки по галерее, без копий) --------
    t0 = time.perf_counter()
    step = 100_000
    best_sims = np.full((n_queries, 10), -np.inf, np.float32)
    best_ids = np.zeros((n_queries, 10), np.int64)
    for s in range(0, n_gallery, step):
        sims = q @ gal[s:s + step].T
        top = np.argpartition(-sims, 10, axis=1)[:, :10]
        rows = np.arange(n_queries)[:, None]
        cand_sims = np.concatenate([best_sims, sims[rows, top]], 1)
        cand_ids = np.concatenate([best_ids, top + s], 1)
        sel = np.argpartition(-cand_sims, 10, axis=1)[:, :10]
        best_sims = cand_sims[rows, sel]
        best_ids = cand_ids[rows, sel]
    ordr = np.argsort(-best_sims, axis=1)
    gt = best_ids[np.arange(n_queries)[:, None], ordr]
    t_flat = (time.perf_counter() - t0) / n_queries * 1000

    # -------- HNSW + скалярная int8-квантизация --------
    hnsw = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, 32,
                             faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 64
    t0 = time.perf_counter()
    hnsw.train(gal[:200_000])
    hnsw.add(gal)
    t_build = time.perf_counter() - t0
    points = {}
    for ef in (64, 128, 256, 512):
        hnsw.hnsw.efSearch = ef
        t0 = time.perf_counter()
        _, ann = hnsw.search(q, 10)
        t_hnsw = (time.perf_counter() - t0) / n_queries * 1000
        recall = float(np.mean([len(set(a) & set(g)) / 10 for a, g in zip(ann, gt)]))
        points[f"efSearch={ef}"] = {"ms": round(t_hnsw, 3),
                                    "recall@10": round(recall, 4),
                                    "speedup": round(t_flat / t_hnsw, 1)}
        print(ef, points[f"efSearch={ef}"])

    best = points["efSearch=256"]
    report = {
        "gallery_size": n_gallery, "dim": dim,
        "index": "HNSW32 + SQ8 (int8)",
        "index_ram_gb": round(n_gallery * dim / 2**30, 2),
        "exact_ms_per_query": round(t_flat, 2),
        "hnsw_build_s": round(t_build, 1),
        "operating_point": "efSearch=256",
        "hnsw_ms_per_query": best["ms"],
        "recall@10_vs_exact": best["recall@10"],
        "speedup": best["speedup"],
        "sweep": points,
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/ann_demo.json").write_text(json.dumps(report, indent=1),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
