"""Продовый экстрактор признака: файл кадра + BBox -> L2-нормированный вектор.

Единый путь и для `submission.csv` / `embeddings.npy`, и для замера скорости
на стенде — чтобы сданные эмбеддинги в точности совпадали с тем, что выдаёт
измеряемый экстрактор (организаторы проверяют, что submission получен
реальной моделью).

Что сделано ради скорости (замер по протоколу стенда, см. src/bench_official.py):

* **Без TTA-отражения.** В паре моделей оно не даёт ничего (mAP@10 0.8828 ->
  0.8827, Δ −0.0000 ± 0.0006 на 30 сидах), а время удваивает.
* **Декодирование JPEG на CPU в пуле потоков**, кроп там же, на GPU уходит
  только кроп; ресайз — на GPU. nvJPEG из torchvision оказался привязан к
  одному ядру (Хаффман-стадия на CPU): 5.1 мс на кадр 1920×1080 и в пакете,
  и поштучно, — это ограничивало пропускную способность ~66 FPS при 4.3 мс
  на сам проход сетей. libjpeg-turbo в 12 потоках — 0.47 мс на кадр.
  Потоки масштабируются по ядрам стенда (os.cpu_count()).
* **CUDA-графы** для фиксированных размеров батча 1/8/16/32: весь проход
  сети записывается один раз и запускается одной командой. На батче 1 это
  снимает накладные расходы на запуск сотен ядер из Python — критично на
  стендовом CPU Xeon 2.0 ГГц. Выход графа совпадает с обычным (cos = 1.0).
  RoPE-эмбеддинги DINOv3 для этого считаются один раз заранее.

Ресайз — бикубический со сглаживанием (antialias), как у torchvision.Resize
на PIL-изображениях при обучении: без сглаживания уменьшение кропа ~600 px до
256–320 даёт алиасинг и меняет признак.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .ensemble import load_manifest
from .infer import load_model

PAD = 0.06                                     # как в src/prepare_crops.py
GRAPH_BATCHES = (1, 8, 16, 32)
PIPE_CHUNK = 8                                 # порция конвейера декод/сеть
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _freeze_rope(model, size):
    rope = getattr(model.backbone, "rope", None)
    if rope is None:
        return
    p = model.backbone.patch_embed.patch_size
    cached = rope.get_embed(shape=(size // p[0], size // p[1]))
    rope.get_embed = lambda shape=None, _c=cached: _c


class _Member:
    """Одна модель состава с набором CUDA-графов под размеры батча."""

    def __init__(self, spec, device, graphs=True):
        c = Config()
        c.backbone, c.img_size = spec["backbone"], spec["img_size"]
        self.model = load_model(spec["ckpt"], c, device)
        self.size = spec["img_size"]
        self.weight = float(spec["weight"])
        self.graphs = {}
        if graphs and device == "cuda":
            _freeze_rope(self.model, self.size)
            for bs in GRAPH_BATCHES:
                self.graphs[bs] = self._capture(bs, device)

    def _forward(self, x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             cache_enabled=False):
            return self.model.extract(x, flip_tta=False)

    def _capture(self, bs, device):
        x = torch.zeros(bs, 3, self.size, self.size, device=device)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = self._forward(x)
        return g, x, y

    def __call__(self, x):
        n = x.shape[0]
        bs = next((b for b in GRAPH_BATCHES if b >= n), None)
        if bs is None or bs not in self.graphs:
            return self._forward(x).float()
        g, xs, ys = self.graphs[bs]
        xs[:n].copy_(x)
        if n < bs:
            xs[n:].zero_()
        g.replay()
        return ys[:n].float().clone()


class Extractor:
    def __init__(self, manifest=None, device="cuda", graphs=True, gpu_decode=False):
        self.device = device
        self.members = [_Member(m, device, graphs) for m in load_manifest(manifest)]
        self.mean, self.std = MEAN.to(device), STD.to(device)
        self.gpu_decode = gpu_decode
        self.pool = ThreadPoolExecutor(max(1, min(16, os.cpu_count() or 1)))

    @staticmethod
    def _box(img, box):
        x, y, w, h = box
        H, W = img.shape[1:]
        px, py = w * PAD, h * PAD
        x0, y0 = max(0, int(x - px)), max(0, int(y - py))
        x1, y1 = min(W, int(x + w + px)), min(H, int(y + h + py))
        return img[:, y0:y1, x0:x1]

    def _one(self, item):
        from torchvision.io import ImageReadMode, decode_jpeg, read_file
        path, box = item
        img = decode_jpeg(read_file(str(path)), mode=ImageReadMode.RGB)
        return self._box(img, box).contiguous().to(self.device)

    def _crops(self, items):
        if self.gpu_decode:                        # прежний путь (nvJPEG)
            from torchvision.io import decode_jpeg, read_file
            raw = [read_file(str(p)) for p, _ in items]
            imgs = decode_jpeg(raw, device=self.device)
            return [self._box(img, box) for img, (_, box) in zip(imgs, items)]
        if len(items) == 1:
            return [self._one(items[0])]
        return list(self.pool.map(self._one, items))

    def _embed(self, crops):
        feats = []
        for m in self.members:
            batch = torch.cat([
                F.interpolate(c[None].float(), size=(m.size, m.size), mode="bicubic",
                              align_corners=False, antialias=True) for c in crops])
            batch = (batch.clamp_(0, 255) / 255.0 - self.mean) / self.std
            feats.append(m.weight * F.normalize(m(batch), dim=1))
        return F.normalize(torch.cat(feats, 1), dim=1)

    @torch.no_grad()
    def extract(self, items):
        """items: список (путь к кадру, (x, y, w, h)). Возвращает (B, D) float32."""
        if self.gpu_decode or len(items) < 2 * PIPE_CHUNK:
            return self._embed(self._crops(items)).cpu().numpy()
        # конвейер: пока GPU считает порцию (запуск асинхронный), пул потоков
        # уже декодирует следующую — время ≈ max(декод, сеть), а не сумма
        futs = [self.pool.submit(self._one, it) for it in items]
        out = [self._embed([f.result() for f in futs[i:i + PIPE_CHUNK]])
               for i in range(0, len(items), PIPE_CHUNK)]
        return torch.cat(out).cpu().numpy()

    def extract_frame(self, df, images_dir, batch_size=32):
        """Эмбеддинги для всех строк CSV (image_id, x, y, w, h), порядок строк сохраняется."""
        images_dir = Path(images_dir)
        items = [(images_dir / f"{r.image_id}.jpg", (r.x, r.y, r.w, r.h))
                 for r in df.itertuples()]
        out = [self.extract(items[i:i + batch_size])
               for i in range(0, len(items), batch_size)]
        return np.concatenate(out).astype(np.float32)
