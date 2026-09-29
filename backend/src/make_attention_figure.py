"""Панель attention-карт: подтверждение, что модель опирается на кузов,
оптику и детали, а не на зону (размытого) номера.

  python -m src.make_attention_figure --ckpt runs/holdout/best.pt [--cpu]
"""
import argparse
import io as _io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    if a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from service.engine import ReIDEngine  # noqa: E402
    from src.config import Config

    cfg = Config()
    # лёгкая инициализация движка без галереи: соберём вручную
    import torch
    from src.model import ReIDModel
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    model = ReIDModel(cfg.backbone, ck["num_classes"], 768, pretrained=False,
                      img_size=cfg.img_size).to(device).eval()
    model.load_state_dict(ck["model"])

    eng = ReIDEngine.__new__(ReIDEngine)          # без __init__ (галерея не нужна)
    eng.model, eng.device = model, device
    from torchvision import transforms as T
    from service.engine import IMG_SIZE, MEAN, STD
    eng.tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE),
                                 interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor(), T.Normalize(MEAN, STD)])

    q = pd.read_csv(cfg.data_root / "test_query.csv").head(a.n)
    panels = []
    for _, r in q.iterrows():
        crop = Image.open(cfg.crops_dir / f"{r.image_id}.jpg").convert("RGB")
        png = eng.attention_map(crop)
        overlay = Image.open(_io.BytesIO(png))
        base = crop.resize((IMG_SIZE, IMG_SIZE))
        pair = Image.new("RGB", (IMG_SIZE * 2 + 6, IMG_SIZE), "#1C1D22")
        pair.paste(base, (0, 0))
        pair.paste(overlay, (IMG_SIZE + 6, 0))
        panels.append(pair)

    W = max(p.width for p in panels)
    H = sum(p.height for p in panels) + 8 * (len(panels) - 1)
    canvas = Image.new("RGB", (W, H), "#1C1D22")
    y = 0
    for p in panels:
        canvas.paste(p, (0, y))
        y += p.height + 8
    out = Path("artifacts/figures/attention_proof.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
