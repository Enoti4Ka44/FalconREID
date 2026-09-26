"""Конвертация внешних ReID-датасетов в наш формат.

Выход: parquet с колонками image_path (абсолютный), vehicle_id (строка с
префиксом источника — ID разных датасетов не пересекаются), camera_id
(тоже с префиксом), источник. BBox не нужен: у внешних наборов изображения
уже являются кропами ТС — они подаются в обучение как готовые кропы.

Поддержка:
  * VeRi-формат (VeRi-776 и CARLA-VeRi от sekilab): имена файлов
    вида 0002_c002_00030600_0.jpg -> id=0002, cam=c002
  * VRIC: списки vric_train.txt (имя id cam)
  * VRAI: images_train + train_annotation (json/pkl с vehicle id)
  * BoxCars116k: json/pkl-аннотация, identity per track

Запуск: python -m src.external_data --root "E:/Задание/external" --out artifacts/external_index.parquet
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

VERI_RE = re.compile(r"^(\d+)_c(\d+)")
CARLA_RE = re.compile(r"^\d{14}_(\d+)_(\d+)\.jpg$")   # <ts>_<vid>_<cam>.jpg


def scan_veri_format(img_dir: Path, source: str):
    rows = []
    for p in img_dir.rglob("*.jpg"):
        m = VERI_RE.match(p.name)
        if not m:
            continue
        rows.append({"image_path": str(p), "vehicle_id": f"{source}_{m.group(1)}",
                     "camera_id": f"{source}_c{m.group(2)}", "source": source})
    return rows


def scan_carla(root: Path):
    rows = []
    for p in root.rglob("*.jpg"):
        m = CARLA_RE.match(p.name)
        if not m:
            continue
        rows.append({"image_path": str(p), "vehicle_id": f"carla_{m.group(1)}",
                     "camera_id": f"carla_c{m.group(2)}", "source": "carla"})
    return rows


VRIC_CAM_RE = re.compile(r"^(MVI_\d+)")


def scan_vric(root: Path):
    """Kaggle-раскладка: vric/<vehicle_id>/MVI_<видео>_<трек>_imgNNN.jpg —
    номер видео служит камерой/сценой."""
    rows = []
    base = next(root.rglob("vric"), None)
    if base is None:
        return rows
    if (base / "vric").is_dir():      # Windows: регистронезависимый rglob ловит внешний VRIC/
        base = base / "vric"
    for id_dir in base.iterdir():
        if not id_dir.is_dir():
            continue
        for p in id_dir.glob("*.jpg"):
            m = VRIC_CAM_RE.match(p.name)
            cam = m.group(1) if m else "unk"
            rows.append({"image_path": str(p), "vehicle_id": f"vric_{id_dir.name}",
                         "camera_id": f"vric_{cam}", "source": "vric"})
    return rows


def scan_boxcars(root: Path):
    """BoxCars116k: dataset.pkl -> samples (идентичность) -> instances (кадры).
    Кадры уже вырезаны с запасом; идентичность видна одной камерой (трек)."""
    import pickle
    pkl = root / "dataset.pkl"
    if not pkl.exists():
        return []
    d = pickle.load(open(pkl, "rb"), encoding="latin-1")
    img_root = root / "images"
    rows = []
    for s in d["samples"]:
        vid = f"boxcars_{s['id']}"
        cam = f"boxcars_{s.get('camera', 'unk')}"
        for inst in s.get("instances", []):
            p = img_root / inst["path"]
            if p.exists():
                rows.append({"image_path": str(p), "vehicle_id": vid,
                             "camera_id": cam, "source": "boxcars"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:/Задание/external")
    ap.add_argument("--out", default="artifacts/external_index.parquet")
    a = ap.parse_args()
    root = Path(a.root)

    rows = []
    # --- CARLA-VeRi (sekilab), полностью открытая синтетика ---
    carla = next(root.rglob("VeRi_CARLA_dataset"), None) or next(
        (d for d in root.iterdir() if d.is_dir() and "CARLA" in d.name.upper()), None)
    if carla:
        n0 = len(rows)
        rows += scan_carla(carla)
        print(f"carla: +{len(rows) - n0}")

    # --- VeRi-776 (зеркало) ---
    for cand in root.rglob("image_train"):
        if "veri" in str(cand).lower() and "carla" not in str(cand).lower():
            n0 = len(rows)
            rows += scan_veri_format(cand.parent, "veri")
            print(f"veri-776: +{len(rows) - n0}")
            break

    # --- VRIC ---
    n0 = len(rows)
    rows += scan_vric(root)
    print(f"vric: +{len(rows) - n0}")

    # --- BoxCars116k ---
    bc = next(root.rglob("BoxCars116k"), None)
    if bc and bc.is_dir():
        n0 = len(rows)
        rows += scan_boxcars(bc)
        print(f"boxcars: +{len(rows) - n0}")

    df = pd.DataFrame(rows)
    if len(df):
        # идентичности с 4+ снимками (K=4 в PK-сэмплере); кросс-камерность
        # желательна (carla), но одно-камерные треки (vric) тоже полезны —
        # они расширяют классификационную голову ночью/блюром
        ok = (df.groupby("vehicle_id")["image_path"].count() >= 4)
        keep = set(ok[ok].index)
        df = df[df.vehicle_id.isin(keep)].reset_index(drop=True)
    Path(a.out).parent.mkdir(exist_ok=True)
    df.to_parquet(a.out)
    if len(df):
        stats = df.groupby("source").agg(imgs=("image_path", "count"),
                                         ids=("vehicle_id", "nunique"))
        print(stats)
    print(f"итого: {len(df)} изображений, {df.vehicle_id.nunique() if len(df) else 0} идентичностей -> {a.out}")


if __name__ == "__main__":
    main()
