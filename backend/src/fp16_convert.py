"""Конверсия чекпойнтов ансамбля в fp16-хранение.

Инференс и так идёт в autocast fp16 — хранение весов в fp16 не меняет
качества (проверяется ниже), но вдвое сокращает размер файлов и
освобождает лимит 2048 МБ под дополнительную модель.

Запуск: python -m src.fp16_convert [--check]
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .ensemble import load_manifest


def convert(ckpt_path: Path, backup_dir: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if all(v.dtype != torch.float32 for v in sd.values() if v.is_floating_point()):
        print(f"{ckpt_path.name}: уже fp16, пропуск")
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    bak = backup_dir / ckpt_path.name
    if not bak.exists():
        shutil.copy2(ckpt_path, bak)
    ck["model"] = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(ck, ckpt_path)
    mb = ckpt_path.stat().st_size / 2**20
    print(f"{ckpt_path.name}: -> fp16, {mb:.0f} МБ (бэкап: {bak})")


def check(manifest, device="cpu", n=8):
    """Косинус эмбеддингов fp16-чекпойнта против fp32-бэкапа на реальных кропах."""
    import pandas as pd
    from .dataset import CropDataset, build_transforms
    from .infer import load_model
    cfg0 = Config()
    df = pd.read_csv(cfg0.data_root / "test_query.csv").head(n)
    for m in manifest:
        cfg = Config()
        cfg.backbone, cfg.img_size = m["backbone"], m["img_size"]
        ds = CropDataset(df, cfg.crops_dir,
                         build_transforms(cfg.img_size, cfg.mean, cfg.std, train=False),
                         with_labels=False)
        x = torch.stack([ds[i][0] for i in range(n)]).to(device)
        embs = []
        for path in (m["ckpt"], f"runs/fp32_backup/{Path(m['ckpt']).name}"):
            model = load_model(path, cfg, device)
            with torch.no_grad():
                e = model.extract(x).float().cpu().numpy()
            embs.append(e / np.linalg.norm(e, axis=1, keepdims=True))
            del model
        cos = (embs[0] * embs[1]).sum(1)
        print(f"{Path(m['ckpt']).name}: cos(fp16, fp32) min={cos.min():.6f} mean={cos.mean():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="сверить эмбеддинги с fp32-бэкапом")
    a = ap.parse_args()
    manifest = load_manifest()
    backup = Path("runs/fp32_backup")
    for m in manifest:
        convert(Path(m["ckpt"]), backup)
    total = sum(Path(m["ckpt"]).stat().st_size for m in manifest) / 2**20
    print(f"суммарно веса ансамбля: {total:.0f} МБ (лимит 2048)")
    if a.check:
        check(manifest)


if __name__ == "__main__":
    main()
