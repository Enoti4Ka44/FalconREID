"""Полный инференс одной командой (для проверки жюри).

  python -m src.run_all --data-root /data --out-dir artifacts

Шаги: эмбеддинги query+gallery продовым экстрактором (полный кадр + BBox ->
вектор, src/extractor.py) -> submission.csv + embeddings.npy + candidates.csv.
Без доступа в интернет. Состав ансамбля — models/ensemble.json, порог отказа —
models/threshold.json (подобран и обоснован на валидации), его можно
переопределить флагом --threshold.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    # При наличии models/ensemble.json состав берётся из манифеста, а --ckpt
    # нужен только чтобы найти рядом threshold.json и postproc.json.
    ap.add_argument("--ckpt", default="models/ensemble.json",
                    help="используется только без манифеста; иначе задаёт каталог "
                         "с threshold.json и postproc.json")
    ap.add_argument("--ckpt2", default="",
                    help="вторая модель при работе без манифеста")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()

    os.environ["DATA_ROOT"] = str(a.data_root)
    data_root = Path(a.data_root)
    for name in ("test_query.csv", "test_gallery.csv"):
        if not (data_root / name).exists():
            sys.exit(f"нет {data_root / name}: укажите каталог данных (--data-root "
                     f"или DATA_DIR для docker compose)")

    # Кэш кропов нужен только старому пути без манифеста (src.train.extract_embeddings);
    # продовый экстрактор читает полные кадры с диска.
    from .ensemble import MANIFEST
    if not MANIFEST.exists():
        # config читает DATA_ROOT при импорте — импортируем после установки env
        from .prepare_crops import main as prepare_crops
        crops = data_root / "crops"
        n_expected = sum(sum(1 for _ in open(data_root / n, encoding="utf-8")) - 1
                         for n in ("test_query.csv", "test_gallery.csv"))
        have = len(list(crops.glob("*.jpg"))) if crops.exists() else 0
        if have < n_expected:
            # каталог данных может быть read-only — кэшируем рядом с артефактами
            try:
                crops.mkdir(parents=True, exist_ok=True)
                test_ok = crops / ".write_test"
                test_ok.touch(); test_ok.unlink()
            except OSError:
                crops = Path(a.out_dir) / "crops"
                os.environ["CROPS_DIR"] = str(crops)
            print(f"кэширование кропов -> {crops}")
            prepare_crops(data_root, crops)

    thr = a.threshold
    if thr is None and os.environ.get("REFUSAL_THRESHOLD"):
        thr = float(os.environ["REFUSAL_THRESHOLD"])   # то же переопределение, что у сервиса
    if thr is None:
        thr_file = Path(a.ckpt).parent / "threshold.json"
        thr = (json.loads(thr_file.read_text(encoding="utf-8"))["threshold"]
               if thr_file.exists() else 0.455)
    print(f"порог режима отказа: {thr}")

    from .infer import main as infer_main
    sys.argv = ["infer", "--ckpt", a.ckpt, "--threshold", str(thr), "--out-dir", a.out_dir]
    if a.ckpt2 and Path(a.ckpt2).exists():
        sys.argv += ["--ckpt2", a.ckpt2]
    infer_main()


if __name__ == "__main__":
    main()
