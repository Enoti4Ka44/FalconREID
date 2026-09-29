"""Однократное кэширование кропов ТС на диск.

Полные кадры весят ~0.7 МБ; вырезав BBox (с небольшим запасом) и ограничив
длинную сторону, мы ускоряем обучение и инференс на порядок.
Имя кропа = image_id (image_id уникален: один кадр — одно размеченное ТС).
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

# Запас вокруг BBox. Больший запас нужен, если при обучении применяется
# сдвиговая аугментация: иначе Pad+RandomCrop достраивает края повтором
# пикселей вместо настоящего контекста сцены.
PAD = float(os.environ.get("BBOX_PAD", "0.06"))
MAX_SIDE = 640    # больше не нужно: сеть работает на 336x336


def crop_one(args):
    img_path, x, y, w, h, out_path = args
    if Path(out_path).exists():
        return
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    px, py = w * PAD, h * PAD
    x0 = max(0, int(x - px)); y0 = max(0, int(y - py))
    x1 = min(W, int(x + w + px)); y1 = min(H, int(y + h + py))
    im = im.crop((x0, y0, x1, y1))
    if max(im.size) > MAX_SIDE:
        s = MAX_SIDE / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
    im.save(out_path, "JPEG", quality=95)


def main(data_root: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for csv_name in ["train.csv", "test_query.csv", "test_gallery.csv"]:
        p = data_root / csv_name
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            jobs.append((
                str(data_root / "images" / f"{r['image_id']}.jpg"),
                int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]),
                str(out_dir / f"{r['image_id']}.jpg"),
            ))
    print(f"cropping {len(jobs)} images -> {out_dir}")
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, _ in enumerate(ex.map(crop_one, jobs, chunksize=64)):
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(jobs)}")
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "data")))
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("DATA_ROOT", "data")) / "crops")
    a = ap.parse_args()
    main(a.data_root, a.out)
