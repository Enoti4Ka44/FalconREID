"""Полный инференс одной командой (для проверки жюри).

  python -m src.run_all --ckpt models/final.pt --data-root /data --out-dir artifacts

Шаги: кэширование кропов (если нет) -> эмбеддинги query+gallery ->
submission.csv + embeddings.npy + candidates.csv. Без доступа в интернет.
Порог отказа берётся из models/threshold.json (подобран и обоснован на валидации),
его можно переопределить флагом --threshold.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    # При наличии models/ensemble.json состав берётся из манифеста, а --ckpt
    # нужен только чтобы найти рядом threshold.json и postproc.json. Раньше
    # здесь стояло имя несуществующего файла, что сбивало с толку.
    ap.add_argument("--ckpt", default="models/ensemble.json",
                    help="используется только без манифеста; иначе задаёт каталог "
                         "с threshold.json и postproc.json")
    ap.add_argument("--ckpt2", default="",
                    help="вторая модель при работе без манифеста")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "E:/Задание/data"))
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()

    os.environ["DATA_ROOT"] = str(a.data_root)
    # config читает DATA_ROOT при импорте — импортируем после установки env
    from .prepare_crops import main as prepare_crops
    data_root = Path(a.data_root)
    crops = data_root / "crops"
    n_expected = 0
    for name in ["test_query.csv", "test_gallery.csv"]:
        n_expected += sum(1 for _ in open(data_root / name)) - 1
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
    if thr is None:
        thr_file = Path(a.ckpt).parent / "threshold.json"
        thr = json.loads(thr_file.read_text())["threshold"] if thr_file.exists() else 0.377
    print(f"порог режима отказа: {thr}")

    from .infer import main as infer_main
    sys.argv = ["infer", "--ckpt", a.ckpt, "--threshold", str(thr), "--out-dir", a.out_dir]
    if a.ckpt2 and Path(a.ckpt2).exists():
        sys.argv += ["--ckpt2", a.ckpt2]
    infer_main()


if __name__ == "__main__":
    main()
