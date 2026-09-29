"""Фигура с примерами из ТЕСТОВОЙ выборки для анализа ошибок (ТЗ §11).

Для каждого примера — запрос и топ-4 галереи по близости из embeddings.npy с
пометками: «та же сцена» (кадр-дубль той же камеры, отсеивается в
candidates.csv), «принят» (≥ τ, другая камера) или «ниже τ».

    python -m src.make_test_examples --data-root <DATA_DIR> \
        --out artifacts/figures/test_examples_v4.png
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import find_image

# (query_id, подпись) — отобраны по embeddings.npy / test_cameras.csv:
# крупные кропы, по одному на каждый характерный случай
EXAMPLES = [
    ("f94a7e7419784d4ebda0ff5f8ef3c9f5",
     "Автобусы (редкий класс): дубль камеры отсеян, верный автобус 0.906 — но чужие автобусы тоже выше порога"),
    ("5fe3bf5786b54f9f8d85861573fc9d2f",
     "Одна марка и модель: верный кадр 0.863 первый, но другой Mercedes V-класса (0.586) тоже принят"),
    ("37cd8f1e7c334f42965941e69fac45d7",
     "Близнецы у порога: 0.466 и 0.460 — модель их не различает, решение за оператором"),
    ("a1022ebe11dc4f2888d2400773e35559",
     "Отказ: максимум 0.404 < порога 0.455 — та же модель Nissan, но другого цвета; ложной находки нет"),
]
TH = 210                                   # высота миниатюры, px


def font(size):
    for p in (Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts/Montserrat-SemiBold.ttf",
              Path("C:/Windows/Fonts/Montserrat-SemiBold.ttf"),
              Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def crop(data_root, df, iid, pad=0.06):
    r = df.loc[iid]
    im = Image.open(find_image(data_root / "images", iid)).convert("RGB")
    px, py = r.w * pad, r.h * pad
    im = im.crop((max(0, int(r.x - px)), max(0, int(r.y - py)),
                  min(im.width, int(r.x + r.w + px)), min(im.height, int(r.y + r.h + py))))
    s = TH / im.height
    im = im.resize((max(1, int(im.width * s)), TH))
    return im.crop((0, 0, min(im.width, 300), TH))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "data")))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/figures/test_examples_v4.png"))
    a = ap.parse_args()

    q = pd.read_csv(a.data_root / "test_query.csv", dtype={"image_id": str})
    g = pd.read_csv(a.data_root / "test_gallery.csv", dtype={"image_id": str})
    emb = np.load(a.artifacts / "embeddings.npy")
    sim = emb[:len(q)] @ emb[len(q):].T
    cams = pd.read_csv(a.artifacts / "test_cameras.csv", dtype={"image_id": str})
    grp = dict(zip(cams.image_id, cams.camera_group))
    tau = json.loads(Path("models/threshold.json").read_text(encoding="utf-8"))["threshold"]
    qi = {v: i for i, v in enumerate(q.image_id)}
    qdf, gdf = q.set_index("image_id"), g.set_index("image_id")

    f1, f2 = font(26), font(22)
    rh = TH + 80
    out = Image.new("RGB", (1900, rh * len(EXAMPLES)), "white")
    d = ImageDraw.Draw(out)
    for row, (qid, caption) in enumerate(EXAMPLES):
        y = row * rh
        d.text((10, y + 6), caption, font=f1, fill=(49, 15, 83))
        i = qi[qid]
        x = 10
        im = crop(a.data_root, qdf, qid)
        out.paste(im, (x, y + 42))
        d.rectangle([x, y + 42, x + im.width, y + 42 + TH], outline=(255, 0, 83), width=5)
        d.text((x + 6, y + 48), "ЗАПРОС", font=f2, fill="white", stroke_width=2, stroke_fill="black")
        x += im.width + 40
        for j in np.argsort(-sim[i])[:4]:
            gid = g.image_id[j]
            im = crop(a.data_root, gdf, gid)
            out.paste(im, (x, y + 42))
            same = grp[qid] == grp[gid]
            s = float(sim[i, j])
            ok = s >= tau and not same
            color = (0, 160, 90) if ok else ((150, 150, 150) if same else (220, 120, 0))
            d.rectangle([x, y + 42, x + im.width, y + 42 + TH], outline=color, width=5)
            label = f"{s:.3f} · " + ("та же сцена" if same else ("принят" if ok else "ниже порога"))
            d.text((x + 6, y + 42 + TH - 32), label, font=f2, fill="white",
                   stroke_width=3, stroke_fill="black")
            x += im.width + 18
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(a.out)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
