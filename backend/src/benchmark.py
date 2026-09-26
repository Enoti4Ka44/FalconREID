"""Замер производительности: латентность (батч=1) и пропускная способность (FPS).

Методология повторяет порядок проверки организаторов: прогрев, затем
стабильные замеры на реальных кропах тестовой выборки.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import CropDataset, build_transforms
from .infer import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="не нужен при наличии models/ensemble.json")
    ap.add_argument("--ckpt2", default=None, help="вторая модель ансамбля")
    ap.add_argument("--backbone2", default="convnext_base.fb_in22k_ft_in1k")
    ap.add_argument("--img-size2", type=int, default=224)
    ap.add_argument("--n-latency", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from .ensemble import MANIFEST, load_manifest
    if MANIFEST.exists():
        models = []
        for m in load_manifest():
            c = Config()
            c.backbone, c.img_size = m["backbone"], m["img_size"]
            models.append((load_model(m["ckpt"], c, device), c))
        print(f"бенчмарк ансамбля из {len(models)} моделей (manifest)")
    else:
        models = [(load_model(a.ckpt, cfg, device), cfg)]
        if a.ckpt2:
            cfg2 = Config()
            cfg2.backbone, cfg2.img_size = a.backbone2, a.img_size2
            models.append((load_model(a.ckpt2, cfg2, device), cfg2))
    df = pd.read_csv(cfg.data_root / "test_query.csv")
    datasets = [CropDataset(df, c.crops_dir,
                            build_transforms(c.img_size, c.mean, c.std, train=False),
                            with_labels=False) for _, c in models]

    def extract_all(i):
        for (m, _), ds in zip(models, datasets):
            x = ds[i][0].unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
                m.extract(x)

    # -------- латентность, батч = 1 (полный цикл: препроцессинг + модели) --------
    for i in range(10):
        extract_all(i)
    torch.cuda.synchronize() if device == "cuda" else None
    times = []
    for i in range(a.n_latency):
        t0 = time.perf_counter()
        extract_all(i % len(df))
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    lat = {"mean_ms": float(np.mean(times)), "p50_ms": float(np.median(times)),
           "p95_ms": float(np.percentile(times, 95))}

    # -------- пропускная способность, пакетная обработка --------
    # два прохода: первый прогревает воркеры DataLoader и кэш ОС,
    # замер по второму (иначе спавн процессов Windows доминирует в числах)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    loaders = [DataLoader(ds, batch_size=a.batch, num_workers=c.num_workers,
                          pin_memory=True, persistent_workers=True)
               for (_, c), ds in zip(models, datasets)]
    n = dt = 0
    for rep in range(2):
        n, t0 = 0, time.perf_counter()
        for (m, _), dl in zip(models, loaders):
            with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
                for img, _, _ in dl:
                    m.extract(img.to(device, non_blocking=True))
                    n += len(img)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    # изображений в СЕКУНДУ сквозь весь ансамбль: каждый кадр проходит все модели
    n_frames = n // len(models)
    thr = {"images": n_frames, "seconds": round(dt, 2),
           "fps": round(n_frames / dt, 1), "batch": a.batch,
           "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
           if device == "cuda" else None}

    report = {"device": device, "models": len(models),
              "latency_bs1": lat, "throughput": thr,
              "note": "латентность включает препроцессинг; TTA flip; ансамбль из "
                      f"{len(models)} модел(ей)"}
    print(json.dumps(report, indent=1, ensure_ascii=False))
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/benchmark.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
