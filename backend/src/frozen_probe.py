"""Отбор кандидатов в ансамбль без дообучения backbone.

Backbone заморожен, признаки для всех кропов считаются один раз и кэшируются;
поверх них учится только лёгкая голова (Linear -> BNNeck -> CosFace + triplet).
Это даёт ЧЕСТНЫЙ замер (голова видит лишь 1233 обучающих ID, метрики — на 308
отложенных) и стоит минуты вместо часов, поэтому годится для перебора крупных
backbone, которые дообучить на 16 ГБ нельзя.

Этап 1:  python -m src.frozen_probe extract
Этап 2:  python -m src.frozen_probe head
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .dataset import CropDataset, build_transforms, build_val_protocol, split_train_val
from .eval_utils import compute_reid_metrics
from .losses import batch_hard_triplet
from .model import MarginClassifier
from .postproc_experiments import aqe, dba, norm
from .quad_eval import refusal_at

CACHE = Path("artifacts/frozen")

# tag -> (backbone timm, размер входа)
CANDIDATES = {
    # калибровка: те же backbone, что в проде, но без дообучения
    "cal_vitl_dinov2":  ("vit_large_patch14_reg4_dinov2.lvd142m", 336),
    "cal_dinov3l":      ("vit_large_patch16_dinov3.lvd1689m", 256),
    # крупные кандидаты
    "dinov3_hplus":     ("vit_huge_plus_patch16_dinov3.lvd1689m", 256),
    "dinov2_giant":     ("vit_giant_patch14_reg4_dinov2.lvd142m", 252),
    "siglip_so400m":    ("vit_so400m_patch16_siglip_384", 384),
    "eva02_l448":       ("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", 448),
    # замена слабого ConvNeXt: та же архитектура, но предобучка DINOv3
    "convnextl_dinov3": ("convnext_large.dinov3_lvd1689m", 256),
    "convnextb_dinov3": ("convnext_base.dinov3_lvd1689m", 224),
    # вторая волна: ДРУГИЕ семейства предобучения. Замеры первой волны
    # показали, что в ансамбле выигрывает декоррелированность, а не размер,
    # поэтому здесь собраны иные данные и иные целевые функции:
    # CLIP/SigLIP — контраст «изображение-текст», EVA02/BEiTv2 — masked image
    # modeling, ConvNeXt/Swin in22k — обычная supervised-классификация.
    "clip_vitl336":     ("vit_large_patch14_clip_336.openai", 336),
    "convnextl_in22k":  ("convnext_large.fb_in22k_ft_in1k", 288),
    "swinl_in22k":      ("swin_large_patch4_window12_384.ms_in22k_ft_in1k", 384),
    "beitv2_l":         ("beitv2_large_patch16_224.in1k_ft_in22k_in1k", 224),
    "eva02_b448":       ("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", 448),
    "caformer_b36":     ("caformer_b36.sail_in22k_ft_in1k_384", 384),
}


def protocol():
    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    train_df, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    vq, vg = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    return cfg, train_df, vq, vg


@torch.no_grad()
def _extract(backbone, cfg, frame, size, bs):
    tf = build_transforms(size, cfg.mean, cfg.std, train=False)
    dl = DataLoader(CropDataset(frame, cfg.crops_dir, tf, with_labels=False),
                    batch_size=bs, num_workers=6, pin_memory=True)
    out = []
    with torch.autocast("cuda", enabled=True):
        for img, _, _ in dl:
            x = img.cuda(non_blocking=True)
            f = backbone(x) + backbone(torch.flip(x, dims=[3]))   # flip-TTA
            out.append(f.float().cpu())
    return torch.cat(out).numpy()


def cmd_extract(only=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg, train_df, vq, vg = protocol()
    for tag, (name, size) in CANDIDATES.items():
        if only and tag not in only:
            continue
        path = CACHE / f"{tag}.npz"
        if path.exists():
            print(f"{tag}: уже посчитан", flush=True)
            continue
        t0 = time.time()
        try:
            kw = dict(pretrained=True, num_classes=0)
            if "vit" in name and "convnext" not in name:
                kw["img_size"] = size
            bb = timm.create_model(name, **kw).cuda().eval()
        except Exception as e:
            print(f"{tag}: НЕ СОЗДАЛСЯ — {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        bs = 16 if size >= 384 else 32
        try:
            tr = _extract(bb, cfg, train_df, size, bs)
            q = _extract(bb, cfg, vq, size, bs)
            g = _extract(bb, cfg, vg, size, bs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(4, bs // 4)
            print(f"{tag}: OOM, повтор с batch={bs}", flush=True)
            tr = _extract(bb, cfg, train_df, size, bs)
            q = _extract(bb, cfg, vq, size, bs)
            g = _extract(bb, cfg, vg, size, bs)
        np.savez(path, train=tr, q=q, g=g)
        n = sum(p.numel() for p in bb.parameters())
        print(f"{tag}: dim={tr.shape[1]} параметров {n/1e6:.0f}М "
              f"fp16 {n*2/2**20:.0f} МиБ / int8 {n*1.06/2**20:.0f} МиБ "
              f"за {time.time()-t0:.0f}с", flush=True)
        del bb
        torch.cuda.empty_cache()


class Head(nn.Module):
    """Лёгкая голова поверх замороженных признаков."""

    def __init__(self, in_dim, num_classes, embed_dim=768, scale=48.0, margin=0.25):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.bn.bias.requires_grad_(False)
        self.cls = MarginClassifier(embed_dim, num_classes, scale, margin)

    def forward(self, x, labels=None):
        f = self.proj(x)
        fb = self.bn(f)
        if labels is None:
            return F.normalize(fb, dim=1)
        return f, fb, self.cls(fb, labels)


def train_head(tr, labels, q, g, num_classes, epochs=100, p=32, k=4, lr=3e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda"
    X = torch.from_numpy(tr).float().to(dev)
    X = F.normalize(X, dim=1)                      # признаки backbone нормируем
    y = torch.from_numpy(labels).long().to(dev)
    head = Head(X.shape[1], num_classes).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    by_id = {}
    for i, lab in enumerate(labels):
        by_id.setdefault(int(lab), []).append(i)
    ids = list(by_id)
    steps = max(1, len(labels) // (p * k))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps,
                                                pct_start=0.1)
    rng = np.random.default_rng(seed)
    head.train()
    for _ in range(epochs):
        for _ in range(steps):
            pick = rng.choice(ids, size=min(p, len(ids)), replace=False)
            idx = []
            for pid in pick:
                pool = by_id[int(pid)]
                idx.extend(rng.choice(pool, size=k, replace=len(pool) < k))
            idx = torch.tensor(idx, device=dev)
            feat, fb, logits = head(X[idx], y[idx])
            loss = (F.cross_entropy(logits, y[idx], label_smoothing=0.1)
                    + batch_hard_triplet(feat.float(), y[idx], 0.3))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    head.eval()
    with torch.no_grad():
        qe = head(F.normalize(torch.from_numpy(q).float().to(dev), dim=1)).cpu().numpy()
        ge = head(F.normalize(torch.from_numpy(g).float().to(dev), dim=1)).cpu().numpy()
    return qe, ge


def cmd_head(only=None, out="runs/frozen_probe.json"):
    cfg, train_df, vq, vg = protocol()
    args = (vq.vehicle_id.values, vg.vehicle_id.values,
            vq.camera_id.values, vg.camera_id.values)
    same = args[2][:, None] == args[3][None, :]
    has = np.array([((args[1] == v) & (args[3] != c)).any()
                    for v, c in zip(args[0], args[2])])
    uniq = sorted(train_df.vehicle_id.unique())
    vid2lab = {v: i for i, v in enumerate(uniq)}
    labels = np.array([vid2lab[v] for v in train_df.vehicle_id])

    results = {}
    for tag in CANDIDATES:
        if only and tag not in only:
            continue
        path = CACHE / f"{tag}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        row = {"dim_raw": int(d["train"].shape[1])}
        # 1) без всякого обучения — сырые признаки backbone
        q0, g0 = norm(d["q"]), norm(d["g"])
        m0 = compute_reid_metrics(q0, g0, *args)
        row["zeroshot_mAP"] = round(m0["mAP"], 4)
        # 2) с лёгкой обученной головой
        qe, ge = train_head(d["train"], labels, d["q"], d["g"], len(uniq))
        gp = dba(norm(ge), k=3)
        qp = aqe(norm(qe), gp, k=1)
        m = compute_reid_metrics(qp, gp, *args)
        r = refusal_at(np.where(same, -1.0, qp @ gp.T), has)
        row.update({"head_mAP": round(m["mAP"], 4), "head_R1": round(m["Rank-1"], 4),
                    "head_mINP": round(m["mINP"], 4), "F1": round(r["F1"], 4),
                    "TNR": round(r["TNR"], 4)})
        np.savez(CACHE / f"head_{tag}.npz", q=qe, g=ge)
        results[tag] = row
        print(f"{tag:18s} zero-shot mAP={row['zeroshot_mAP']:.4f} | "
              f"голова mAP={row['head_mAP']:.4f} R1={row['head_R1']:.4f} "
              f"F1={row['F1']:.3f}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "head"])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.cmd == "extract":
        cmd_extract(a.only)
    else:
        cmd_head(a.only)
