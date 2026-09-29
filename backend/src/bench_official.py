"""Замер производительности строго по протоколу постановщика (Q&A, вопрос 31/34).

В замер входит полный цикл на одно ТС: чтение файла с диска, декодирование,
crop по BBox, preprocessing, forward, postprocessing, L2-нормализация.

* latency_b1 — медиана полного цикла при batch=1, 300 прогонов после 50
  прогревочных, с синхронизацией CUDA до и после каждого замера;
* throughput — устойчивый FPS при батчах 1/8/16/32, прогон не короче 10 с,
  в баллы идёт лучший.

Баллы (20% итога): latency ≤40 мс — полный, 40–80 — линейно, >80 — 0;
FPS ≥100 — полный, 50–100 — линейно, <50 — 0.

Прежний `src/benchmark.py` читал заранее вырезанные кропы, то есть не
учитывал декодирование полного кадра 1920×1080 — а это заметная часть
времени. Этот скрипт меряет то, что будут мерить организаторы.

    python -m src.bench_official                 # текущий манифест
    python -m src.bench_official --no-flip       # без TTA-отражения
    python -m src.bench_official --gpu-decode    # декодирование JPEG на GPU
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .config import Config, find_image
from .ensemble import load_manifest
from .infer import load_model

PAD = 0.06
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def score_latency(ms):
    return 1.0 if ms <= 40 else (0.0 if ms > 80 else (80 - ms) / 40)


def score_fps(fps):
    return 1.0 if fps >= 100 else (0.0 if fps < 50 else (fps - 50) / 50)


class Extractor:
    """Полный конвейер «файл + BBox -> вектор», как его будет вызывать стенд."""

    def __init__(self, device="cuda", flip=True, gpu_decode=False, compile_=False,
                 manifest=None):
        self.device, self.flip, self.gpu_decode = device, flip, gpu_decode
        self.members = []
        for m in load_manifest(manifest):
            c = Config()
            c.backbone, c.img_size = m["backbone"], m["img_size"]
            model = load_model(m["ckpt"], c, device)
            if compile_:
                model = torch.compile(model, mode="max-autotune")
            self.members.append((model, m["img_size"], float(m["weight"])))
        self.mean, self.std = MEAN.to(device), STD.to(device)

    def _load(self, path, box):
        x, y, w, h = box
        if self.gpu_decode:
            from torchvision.io import decode_jpeg, read_file
            img = decode_jpeg(read_file(str(path)), device=self.device)   # (3,H,W) uint8
            H, W = img.shape[1:]
        else:
            pil = Image.open(path).convert("RGB")
            W, H = pil.size
        px, py = w * PAD, h * PAD
        x0, y0 = max(0, int(x - px)), max(0, int(y - py))
        x1, y1 = min(W, int(x + w + px)), min(H, int(y + h + py))
        if self.gpu_decode:
            return img[:, y0:y1, x0:x1]
        crop = np.asarray(pil.crop((x0, y0, x1, y1)))
        return torch.from_numpy(crop).permute(2, 0, 1).to(self.device, non_blocking=True)

    def _load_batch(self, items):
        """Пакетное декодирование: все JPEG батча разом на GPU (nvJPEG)."""
        from torchvision.io import decode_jpeg, read_file
        raw = [read_file(str(p)) for p, _ in items]
        imgs = decode_jpeg(raw, device=self.device)
        crops = []
        for img, (_, (x, y, w, h)) in zip(imgs, items):
            H, W = img.shape[1:]
            px, py = w * PAD, h * PAD
            x0, y0 = max(0, int(x - px)), max(0, int(y - py))
            x1, y1 = min(W, int(x + w + px)), min(H, int(y + h + py))
            crops.append(img[:, y0:y1, x0:x1])
        return crops

    @torch.no_grad()
    def extract(self, items):
        """items: список (путь, (x, y, w, h)). Возвращает (B, D) float32, L2."""
        if self.gpu_decode and len(items) > 1:
            crops = self._load_batch(items)
        else:
            crops = [self._load(p, b) for p, b in items]
        feats = []
        for model, size, weight in self.members:
            batch = torch.stack([
                F.interpolate(c[None].float(), size=(size, size), mode="bicubic",
                              align_corners=False)[0] for c in crops]) / 255.0
            batch = (batch - self.mean) / self.std
            with torch.autocast("cuda", dtype=torch.float16):
                f = model.extract(batch, flip_tta=self.flip)
            feats.append(weight * f.float())
        out = torch.cat(feats, 1)
        return F.normalize(out, dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--gpu-decode", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--n-lat", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--fps-seconds", type=float, default=10.0)
    ap.add_argument("--prod", action="store_true",
                    help="мерить продовый экстрактор src/extractor.py")
    ap.add_argument("--no-graphs", action="store_true",
                    help="продовый экстрактор без CUDA-графов (для сравнения)")
    ap.add_argument("--manifest", default=None,
                    help="другой манифест (для сравнения составов по скорости)")
    ap.add_argument("--out", default="artifacts/bench_official.json")
    a = ap.parse_args()

    cfg = Config()
    df = pd.read_csv(cfg.data_root / "test_query.csv")
    items = [(find_image(cfg.data_root / "images", r.image_id), (r.x, r.y, r.w, r.h))
             for r in df.itertuples()]
    if a.prod:
        # продовый экстрактор — ровно тот код, что строит embeddings.npy
        from .extractor import Extractor as ProdExtractor
        ex = ProdExtractor(manifest=a.manifest, graphs=not a.no_graphs,
                           gpu_decode=a.gpu_decode)
    else:
        ex = Extractor(flip=not a.no_flip, gpu_decode=a.gpu_decode, compile_=a.compile,
                       manifest=a.manifest)

    for i in range(a.warmup):
        ex.extract([items[i % len(items)]])
    times = []
    for i in range(a.n_lat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ex.extract([items[(i + a.warmup) % len(items)]])
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    lat = statistics.median(times)

    fps = {}
    for bs in (1, 8, 16, 32):
        ex.extract(items[:bs])
        n, t0, j = 0, time.perf_counter(), 0
        while time.perf_counter() - t0 < a.fps_seconds:
            batch = [items[(j + k) % len(items)] for k in range(bs)]
            ex.extract(batch)
            n += bs
            j += bs
        torch.cuda.synchronize()
        fps[bs] = n / (time.perf_counter() - t0)
    best_fps = max(fps.values())

    res = {"latency_b1_ms": round(lat, 1), "latency_p95_ms": round(np.percentile(times, 95), 1),
           "fps": {k: round(v, 1) for k, v in fps.items()}, "best_fps": round(best_fps, 1),
           "score_latency": round(score_latency(lat), 3),
           "score_throughput": round(score_fps(best_fps), 3),
           "points_of_20": round(10 * score_latency(lat) + 10 * score_fps(best_fps), 2),
           "prod_extractor": a.prod, "cuda_graphs": a.prod and not a.no_graphs,
           "flip": (not a.no_flip) and not a.prod, "gpu_decode": ex.gpu_decode,
           "compile": a.compile,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
