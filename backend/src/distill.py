"""Дистилляция квада в одну быструю модель (fast-режим сервиса).

Студент (ViT-B) учится воспроизводить 3072-d эмбеддинги ансамбля-учителя
косинусным лоссом на train-кропах. Итог: один forward вместо четырёх
(~35 мс против 141), качество проверяется на замороженном эталоне.

Запуск:
  python -m src.distill                # обучить студента
  python -m src.distill --eval-only    # только оценка чекпойнта
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .dataset import CropDataset, build_transforms, build_val_protocol, split_train_val
from .eval_utils import compute_reid_metrics

TEACHER_DIM = 3072
CKPT = Path("runs/distill/student.pt")


class Student(nn.Module):
    def __init__(self, backbone="vit_base_patch14_reg4_dinov2.lvd142m", img_size=252):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0,
                                          img_size=img_size)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 1024), nn.GELU(),
            nn.Linear(1024, TEACHER_DIM))

    def forward(self, x):
        return F.normalize(self.head(self.backbone(x)), dim=1)

    @torch.no_grad()
    def extract(self, x, flip_tta=True):
        f = self(x)
        if flip_tta:
            f = f + self(torch.flip(x, dims=[3]))
        return F.normalize(f, dim=1)


def teacher_embeddings(df, device, manifest_path):
    from .ensemble import extract_ensemble, load_manifest
    return extract_ensemble(df, device, load_manifest(manifest_path))


def extract_student(model, df, cfg, device, bs=64):
    ds = CropDataset(df, cfg.crops_dir,
                     build_transforms(cfg.img_size, cfg.mean, cfg.std, train=False),
                     with_labels=False)
    dl = DataLoader(ds, batch_size=bs, num_workers=cfg.num_workers, pin_memory=True)
    out = []
    with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
        for img, _, _ in dl:
            out.append(model.extract(img.to(device)).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--teacher", default="models/ensemble_holdout.json",
                    help="манифест учителя. Для честной оценки на эталоне — "
                         "holdout-модели (не видели val-идентичностей); для прода — "
                         "models/ensemble.json")
    ap.add_argument("--tag", default="holdout")
    a = ap.parse_args()
    global CKPT
    CKPT = Path(f"runs/distill/student_{a.tag}.pt")
    cfg = Config()
    cfg.img_size = 252
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.data_root / "train.csv")
    train_df, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    val_q, val_g = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)

    model = Student(img_size=cfg.img_size).to(device)
    if not a.eval_only:
        # --- таргеты учителя (только train-идентичности) ---
        t_path = Path(f"runs/distill/teacher_train_{a.tag}.npy")
        if t_path.exists():
            targets = np.load(t_path)
        else:
            print(f"считаю эмбеддинги учителя ({a.teacher}) на train ...")
            targets = teacher_embeddings(train_df, device, a.teacher)
            np.save(t_path, targets)
        targets_t = torch.tensor(targets, dtype=torch.float32)

        ds = CropDataset(train_df.reset_index(drop=True), cfg.crops_dir,
                         build_transforms(cfg.img_size, cfg.mean, cfg.std, train=True),
                         with_labels=True)
        # индекс кадра нужен для выборки таргета — подменяем метки индексами
        ds.vid2label = {v: i for i, v in enumerate(train_df.vehicle_id)}  # не используется
        idx_ds = list(range(len(ds)))

        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=a.epochs * (len(idx_ds) // a.bs + 1))
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        rng = np.random.default_rng(cfg.seed)

        for ep in range(a.epochs):
            model.train()
            order = rng.permutation(len(idx_ds))
            tot = n = 0
            t0 = time.time()
            for b0 in range(0, len(order), a.bs):
                sel = order[b0:b0 + a.bs]
                imgs = torch.stack([ds[i][0] for i in sel]).to(device, non_blocking=True)
                tgt = F.normalize(targets_t[sel].to(device), dim=1)
                with torch.autocast("cuda", enabled=device == "cuda"):
                    pred = model(imgs)
                    loss = (1 - (pred * tgt).sum(1)).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item() * len(sel); n += len(sel)
            print(f"ep {ep + 1}/{a.epochs} cos_loss={tot / n:.4f} time={time.time() - t0:.0f}s")
        torch.save({"model": model.state_dict()}, CKPT)
    else:
        model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])

    # --- оценка на замороженном эталоне ---
    model.eval()
    q = extract_student(model, val_q, cfg, device)
    g = extract_student(model, val_g, cfg, device)
    from .postproc_experiments import aqe, dba
    g = dba(g, k=3); q = aqe(q, g, k=1)
    m = compute_reid_metrics(q, g, val_q.vehicle_id.values, val_g.vehicle_id.values,
                             val_q.camera_id.values, val_g.camera_id.values)
    print(f"студент: mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} R5={m['Rank-5']:.4f}")
    Path(f"runs/distill/eval_{a.tag}.json").write_text(json.dumps(
        {k: round(float(v), 4) for k, v in m.items()}), encoding="utf-8")
    np.savez(f"artifacts/val_embeddings_student_{a.tag}.npz", q=q, g=g)


if __name__ == "__main__":
    main()
