"""Демонстрация масштабируемости: FAISS HNSW + int8-квантизация на галерее 10^6.

Показывает готовность к городскому масштабу: миллион векторов той же
размерности, что и продовый эмбеддинг (3072-d для v4), скорость запроса,
recall против точного поиска и потребление памяти индекса (SQ8: 1 байт на
компоненту — ~2.9 ГБ на 10^6 × 3072 вместо 11.4 ГБ float32).

Галерея генерируется и добавляется в индекс потоково, чанками: целиком
матрица 10^6 × 3072 float32 (11.4 ГБ) в памяти не держится.

    python -m src.ann_demo                      # 10^6, размерность из artifacts/embeddings.npy
"""
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np


def make_vectors(real, n, rng, dim):
    """Выпуклые комбинации пар РЕАЛЬНЫХ эмбеддингов теста + малый шум: геометрия
    многообразия признаков сохраняется — именно на ней работает HNSW (в отличие
    от i.i.d.-гауссовской синтетики, где графовые индексы вырождаются)."""
    i1 = rng.integers(0, len(real), n)
    i2 = rng.integers(0, len(real), n)
    a = rng.uniform(0.55, 0.95, (n, 1)).astype(np.float32)
    v = a * real[i1] + (1 - a) * real[i2]
    v += 0.03 * rng.standard_normal((n, dim)).astype(np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    faiss.normalize_L2(v)
    return v


def main(n_gallery: int = 1_000_000, n_queries: int = 500, chunk: int = 50_000):
    real = np.load("artifacts/embeddings.npy").astype(np.float32)
    dim = real.shape[1]
    print(f"галерея: {n_gallery:,} векторов dim={dim} "
          f"({n_gallery * dim * 4 / 2**30:.1f} ГБ float32, "
          f"{n_gallery * dim / 2**30:.1f} ГБ int8); основа — {len(real)} реальных векторов")
    q = make_vectors(real, n_queries, np.random.default_rng(7), dim)

    hnsw = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, 32,
                             faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 64
    t0 = time.perf_counter()
    hnsw.train(make_vectors(real, 100_000, np.random.default_rng(1), dim))
    t_build = time.perf_counter() - t0

    # точный топ-10 (ground truth) и HNSW строятся по одним и тем же чанкам
    best_sims = np.full((n_queries, 10), -np.inf, np.float32)
    best_ids = np.zeros((n_queries, 10), np.int64)
    rows = np.arange(n_queries)[:, None]
    t_flat = 0.0
    for s in range(0, n_gallery, chunk):
        g = make_vectors(real, min(chunk, n_gallery - s), np.random.default_rng([42, s]), dim)
        t0 = time.perf_counter()
        sims = q @ g.T
        top = np.argpartition(-sims, 10, axis=1)[:, :10]
        cand_sims = np.concatenate([best_sims, sims[rows, top]], 1)
        cand_ids = np.concatenate([best_ids, top + s], 1)
        sel = np.argpartition(-cand_sims, 10, axis=1)[:, :10]
        best_sims, best_ids = cand_sims[rows, sel], cand_ids[rows, sel]
        t_flat += time.perf_counter() - t0
        t0 = time.perf_counter()
        hnsw.add(g)
        t_build += time.perf_counter() - t0
        if (s // chunk) % 4 == 0:
            print(f"  {s + len(g):,} / {n_gallery:,}", flush=True)
    gt = best_ids[rows, np.argsort(-best_sims, axis=1)]
    t_flat = t_flat / n_queries * 1000

    points = {}
    for ef in (64, 128, 256, 512):
        hnsw.hnsw.efSearch = ef
        t0 = time.perf_counter()
        _, ann = hnsw.search(q, 10)
        t_hnsw = (time.perf_counter() - t0) / n_queries * 1000
        recall = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ann, gt)]))
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
        "cpu_threads": faiss.omp_get_max_threads(),
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/ann_demo.json").write_text(json.dumps(report, indent=1),
                                               encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=500)
    a = ap.parse_args()
    main(a.n, a.queries)
