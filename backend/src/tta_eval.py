"""Многомасштабный TTA: даёт ли усреднение по нескольким входным размерам.

Веса от этого не растут — платим только временем инференса, а запас по
скорости на текущем железе трёхкратный. Меряется на holdout-модели (продовые
модели видели валидационные ТС и дают на них mAP 1.0, для замеров непригодны).

    python -m src.tta_eval --ckpt runs/holdout_b_base/best.pt \
        --backbone vit_base_patch14_reg4_dinov2.lvd142m --native 252 \
        --sizes 196 224 252 280 322
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import (CropDataset, build_transforms, build_val_protocol,
                      split_train_val)
from .eval_utils import compute_reid_metrics
from .model import ReIDModel, ReIDModelParts
from .postproc_experiments import norm

CACHE = Path("artifacts/tta")


def build(ckpt, backbone, native):
    """Модель с интерполяцией позиционных эмбеддингов под любой размер входа."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    cls = ReIDModelParts if any(k.startswith("part_proj.") for k in sd) else ReIDModel
    c = Config()
    c.backbone, c.img_size = backbone, native
    m = cls(backbone, ck["num_classes"], c.embed_dim, pretrained=False, img_size=native)
    if "vit" in backbone or "dinov2" in backbone:
        m.backbone = timm.create_model(backbone, pretrained=False, num_classes=0,
                                       img_size=native, dynamic_img_size=True)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v
                       for k, v in sd.items()})
    return m.cuda().eval(), c


@torch.no_grad()
def extract(m, c, frame, size, bs=32):
    tf = build_transforms(size, c.mean, c.std, train=False)
    dl = DataLoader(CropDataset(frame, c.crops_dir, tf, with_labels=False),
                    batch_size=bs, num_workers=6, pin_memory=True)
    out = []
    with torch.autocast("cuda", enabled=True):
        for img, _, _ in dl:
            out.append(m.extract(img.cuda(non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--native", type=int, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    tag = a.tag or Path(a.ckpt).parent.name

    cfg = Config()
    df = pd.read_csv(cfg.data_root / "train.csv")
    _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    vq0, vg0 = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)

    CACHE.mkdir(parents=True, exist_ok=True)
    pool = {}
    model = None
    for s in a.sizes:
        p = CACHE / f"{tag}_{s}.npz"
        if not p.exists():
            if model is None:
                model, c = build(a.ckpt, a.backbone, a.native)
            np.savez(p, q=extract(model, c, vq0, s), g=extract(model, c, vg0, s))
            print(f"  посчитан {tag}@{s}", flush=True)
        d = np.load(p)
        pool[s] = {**dict(zip(vq0.image_id.values, d["q"])),
                   **dict(zip(vg0.image_id.values, d["g"]))}
    if model is not None:
        del model
        torch.cuda.empty_cache()

    def run(sizes, seeds):
        out = []
        for seed in seeds:
            vq, vg = build_val_protocol(val_df, cfg.distractor_fraction, seed)
            args = (vq.vehicle_id.values, vg.vehicle_id.values,
                    vq.camera_id.values, vg.camera_id.values)
            q = norm(sum(norm(np.stack([pool[s][i] for i in vq.image_id]))
                         for s in sizes))
            g = norm(sum(norm(np.stack([pool[s][i] for i in vg.image_id]))
                         for s in sizes))
            out.append(compute_reid_metrics(q, g, *args)["mAP"])
        return np.array(out)

    seeds = list(range(42, 42 + a.seeds))
    base = run([a.native], seeds)
    print(f"\nодин масштаб {a.native}: mAP={base.mean():.4f}±{base.std():.4f}")
    res = {}
    combos = [[s] for s in a.sizes if s != a.native]
    combos += [sorted(set([a.native, s])) for s in a.sizes if s != a.native]
    combos += [sorted(a.sizes), sorted(a.sizes)[:3], sorted(a.sizes)[-3:]]
    seen = set()
    for c in combos:
        key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        v = run(c, seeds)
        d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        verdict = "ПОДТВЕРЖДЕНО" if abs(d.mean()) > 2 * se else "шум"
        print(f"  {str(c):26s} mAP={v.mean():.4f} Δ={d.mean():+.4f}±{se:.4f} "
              f"({(d > 0).sum()}/{len(d)}) {verdict}")
        res[str(c)] = {"mAP": round(float(v.mean()), 4),
                       "delta": round(float(d.mean()), 4),
                       "se": round(float(se), 4)}
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/tta_{tag}.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
