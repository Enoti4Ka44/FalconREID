"""Пересборка продового решения после того, как рецепт обучения улучшен.

Порядок: обучить финальные модели на ВСЕХ идентичностях -> собрать манифест
ансамбля -> пересчитать порог -> построить индекс галереи -> сформировать
артефакты сдачи -> самопроверка -> бенчмарк.

Holdout-модели нужны только для замеров; в прод идут модели, обученные на
полном train.csv (все 1541 ТС), иначе теряется 20% обучающих данных.

    python -m src.rebuild_prod --plan            # показать, что будет сделано
    python -m src.rebuild_prod --train           # обучить финальные модели
    python -m src.rebuild_prod --assemble        # манифест + порог + артефакты
"""
import os
import argparse
import json
import subprocess
import sys
from pathlib import Path

MODELS = Path("models")
RUNS = Path("runs")

# имя файла весов -> (backbone, размер входа, флаги обучения, вес в конкатенации)
# Состав на текущий момент: ConvNeXt снят (подтверждено на 30 переразбиениях
# протокола: +0.0013 mAP и −171 МиБ). Флаги обучения уточняются по итогам
# абляций рецепта; см. docs/PROGRESS.md.
PROD = {
    # Основная модель выбрана по совокупности «качество + скорость», а не по
    # одному mAP. ViT-H+@320 точнее (mAP@10 0.8828 против 0.8475), но упирается
    # в ~51 FPS даже на RTX 4070 Ti SUPER — это ~10 очков скорости из 20.
    # DINOv3-L с части-признаками в паре даёт 104 FPS и 16 мс — полные 20 очков.
    # Разница по качеству (0.035 mAP@10) стоит ~1.6 очка из 45, по скорости —
    # ~10 из 20. См. docs/PROGRESS.md, итерация 18.
    # + дистилляция структуры сходств от точного, но медленного ViT-H+@320 +
    # ViT-B·части (учитель обучен на тех же данных, эмбеддинги — один раз,
    # src/teacher_emb.py). Вес 1000: KD-лосс ~0.003 на фоне CE ~20. В паре
    # +0.0033 ± 0.0012 mAP@10 на 30 сидах. См. docs/PROGRESS.md, итерация 20.
    "dinov3ps": ("vit_large_patch16_dinov3.lvd1689m", 256,
                 ["--aug", "shift", "--parts", "--grad-ckpt",
                  "--kd", "runs/teacher_prod.npz", "--kd-weight", "1000",
                  "--xbm", "4096"], 1.0),
    # партнёр выбран по декоррелированности: другая архитектура и другая
    # предобучающая выборка (DINOv2 LVD-142M против DINOv3 LVD-1689M)
    # + память по батчам (XBM 4096, с 3-й эпохи): трудные негативы сверх
    # маленького батча. Основная модель +0.0035 ± 0.0005 (среднее 2 сидов
    # против 3), ViT-B +0.005 в паре с каждым из 5 вариантов основной.
    # См. docs/PROGRESS.md, итерация 21.
    "vitparts": ("vit_base_patch14_reg4_dinov2.lvd142m", 252,
                 ["--aug", "shift", "--parts", "--xbm", "4096"], 0.7),
}


def cmd(args, log=None):
    print("  $", " ".join(args[2:] if args[:2] == [sys.executable, "-m"] else args),
          flush=True)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            return subprocess.run(args, stdout=f, stderr=subprocess.STDOUT).returncode
    return subprocess.run(args).returncode


def do_train(holdout=False):
    """holdout=True — модели для ЗАМЕРА (без 20% идентичностей),
    holdout=False — продовые модели (на всех 1541 ТС)."""
    (RUNS / "logs").mkdir(parents=True, exist_ok=True)
    prefix = "hprod" if holdout else "final"
    for name, (backbone, size, extra, _w) in PROD.items():
        run_dir = RUNS / f"{prefix}_{name}"
        if (run_dir / "best.pt").exists() or (run_dir / "final.pt").exists():
            print(f"[{prefix}_{name}] уже обучено")
            continue
        heavy = ["--grad-ckpt"] if any(t in backbone for t in
                                       ("large", "huge", "giant")) else []
        args = [sys.executable, "-m", "src.train", "--run-name", run_dir.name,
                "--backbone", backbone, "--img-size", str(size), *heavy, *extra]
        if holdout:
            args.insert(4, "--holdout-val")
        rc = cmd(args, log=RUNS / "logs" / f"{run_dir.name}.log")
        if rc != 0:
            print(f"[{run_dir.name}] ОБУЧЕНИЕ УПАЛО")
            return False
    return True


def _store(src: Path, dst: Path, fmt: str):
    """Переносит обученный чекпойнт в models/ в формате хранения.

    fp16 — обычное хранение; int8 — упаковка с групповыми масштабами
    (src/int8_pack.py), нужна, когда состав в fp16 не влезает в 2048 МиБ.
    Инференс в обоих случаях идёт в fp16-autocast.
    """
    import torch

    from .int8_pack import pack_checkpoint
    if fmt == "int8":
        before, after = pack_checkpoint(src, dst, 32)
        print(f"  {dst.name}: {before:.0f} -> {after:.0f} МиБ (int8)")
        return
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck["model"] = {k: (v.half() if v.is_floating_point() else v)
                   for k, v in ck["model"].items()}
    torch.save({k: v for k, v in ck.items() if k in ("model", "cfg", "num_classes")}, dst)
    print(f"  {dst.name}: {dst.stat().st_size / 2**20:.0f} МиБ (fp16)")


def do_assemble(data_root, fmt="fp16"):
    models = []
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, (backbone, size, _extra, w) in PROD.items():
        src = RUNS / f"final_{name}" / "final.pt"
        if not src.exists():
            src = RUNS / f"final_{name}" / "best.pt"
        if not src.exists():
            print(f"нет обученной модели {name}: {src}")
            return False
        dst = MODELS / f"final_{name}.pt"
        _store(src, dst, fmt)
        models.append({"ckpt": str(dst).replace("\\", "/"), "backbone": backbone,
                       "img_size": size, "weight": w})
    (MODELS / "ensemble.json").write_text(
        json.dumps({"models": models}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"манифест обновлён: {len(models)} моделей")
    for step in (["src.threshold_official"], ["src.build_gallery"],
                 ["src.run_all", "--data-root", data_root],
                 ["src.validate_submission", "--data-root", data_root],
                 ["src.bench_official", "--prod"]):
        if cmd([sys.executable, "-m", *step]) != 0:
            print(f"ШАГ УПАЛ: {step[0]}")
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--train", action="store_true",
                    help="обучить ПРОДОВЫЕ модели (на всех 1541 ТС)")
    ap.add_argument("--train-holdout", action="store_true",
                    help="обучить модели для ЗАМЕРА тем же рецептом")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    ap.add_argument("--format", default="fp16", choices=["fp16", "int8"],
                    help="формат хранения весов в models/")
    a = ap.parse_args()

    if not PROD:
        print("состав PROD пуст — заполните его по итогам абляций "
              "(docs/PROGRESS.md)")
        return
    if a.plan:
        for name, (bb, size, extra, w) in PROD.items():
            print(f"  {name:12s} {bb} @{size} вес={w} {' '.join(extra)}")
        return
    if a.train_holdout and not do_train(holdout=True):
        return
    if a.train and not do_train():
        return
    if a.assemble:
        if do_assemble(a.data_root, a.format):
            from .ensemble import weights_size_total_mb
            tot, rows = weights_size_total_mb()
            for n, mb, used in rows:
                print(f"  {n:24s} {mb:7.1f} МиБ  {'в ансамбле' if used else 'вне ансамбля'}")
            print(f"  ИТОГО {tot:.0f} МиБ из 2048, свободно {2048 - tot:.0f}")


if __name__ == "__main__":
    main()
