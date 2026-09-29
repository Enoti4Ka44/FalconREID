"""Очередь экспериментов: обучение holdout-модели -> честная оценка на эталоне.

Каждый эксперимент — это (обучить с такими-то флагами) + (померить
`src.eval_candidate`). Очередь возобновляемая: готовые прогоны пропускаются,
поэтому её можно останавливать и запускать заново.

    python -m src.experiments --list
    python -m src.experiments --run d_base d_shift
    python -m src.experiments --run-all
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path("runs")

VITB = "vit_base_patch14_reg4_dinov2.lvd142m"
DINOV3L = "vit_large_patch16_dinov3.lvd1689m"
HPLUS = "vit_huge_plus_patch16_dinov3.lvd1689m"

# имя -> (backbone, размер входа для оценки, доп. флаги train.py)
#
# Этап 1 — быстрые абляции рецепта на ViT-B (эпоха 30 с, прогон ~15 мин).
# Этап 2 — победивший рецепт переносится на крупные модели.
EXPERIMENTS = {
    # --- этап 1: рецепт обучения, дешёвый полигон ---
    "b_base":  (VITB, "252", []),                                  # контроль
    "b_shift": (VITB, "252", ["--aug", "shift"]),
    "b_rrc":   (VITB, "252", ["--aug", "rrc"]),
    "b_ep60":  (VITB, "252", ["--epochs", "60"]),
    "b_322":   (VITB, "322", ["--aug", "shift"]),                  # больше вход
    "b_rect":  (VITB, "280x364", ["--aug", "shift",                # форма кропа 1.3
                                  "--img-h", "280", "--img-size", "364"]),
    # следующие складываются с победившим shift (см. docs/PROGRESS.md)
    "b_ema":   (VITB, "252", ["--aug", "shift", "--ema", "0.999"]),
    "b_llrd":  (VITB, "252", ["--aug", "shift", "--llrd", "0.75"]),
    "b_cam":   (VITB, "252", ["--aug", "shift", "--cam-aware"]),
    "b_s60":   (VITB, "252", ["--aug", "shift", "--epochs", "60"]),
    "b_s90":   (VITB, "252", ["--aug", "shift", "--epochs", "90"]),
    "b_sh10":  (VITB, "252", ["--aug", "shift10"]),                # сдвиг сильнее
    "b_sh15":  (VITB, "252", ["--aug", "shift15"]),
    # Кроп кэширован с запасом 18% вместо 6%. Со сдвиговой аугментацией это
    # бесполезно: длинная сторона всё равно ограничена 640 px, поэтому машина
    # занимает МЕНЬШЕ пикселей. А вот RandomResizedCrop по широкому кропу
    # вырезает настоящие подобласти сцены вместо повтора краевых пикселей.
    "b_pad18": (VITB, "252", ["--aug", "rrc"], "C:/falcon/data/crops_pad18"),
    # по разбору ошибок новой пары: камера с вытянутыми кропами (w/h 2.1)
    # проваливается в 89% случаев, ночная камера — втрое чаще средней
    "bp_lb":    (VITB, "252", ["--aug", "shift", "--parts", "--letterbox"]),
    "bp_night": (VITB, "252", ["--aug", "shift+night", "--parts"]),
    # половина провалов пары — «почти угадал» (зазор < 0.05): нужны трудные
    # негативы, которых в маленьком батче мало. Память по батчам (XBM).
    "bp_xbm":   (VITB, "252", ["--aug", "shift", "--parts", "--xbm", "4096"]),
    "bp_xbm_s2": (VITB, "252", ["--aug", "shift", "--parts", "--xbm", "4096", "--train-seed", "2"]),
    "bp_s2":     (VITB, "252", ["--aug", "shift", "--parts", "--train-seed", "2"]),
    # --- ViT-H+ (841М, 32 блока): подтверждено +0.0188 к ансамблю даже в
    # замороженном виде. Целиком на 16 ГБ не дообучается (одному AdamW нужно
    # 13.5 ГБ), поэтому нижние блоки замораживаются.
    # При замороженном backbone голове нужен LR на порядок выше обычного.
    # backbone заморожен -> память свободна, батч можно взять крупный
    # (на кэшированных признаках P=64 давал +0.0075 против P=32)
    "h_frozen": (HPLUS, "256", ["--aug", "shift", "--freeze-blocks", "-1",
                                "--lr", "2e-3", "--epochs", "60", "--batch-p", "24"]),
    "h_last8":  (HPLUS, "256", ["--aug", "shift", "--freeze-blocks", "24",
                                "--batch-p", "8"]),
    "h_last4":  (HPLUS, "256", ["--aug", "shift", "--freeze-blocks", "28",
                                "--batch-p", "8"]),
    "h_last16": (HPLUS, "256", ["--aug", "shift", "--freeze-blocks", "16",
                                "--batch-p", "6"]),
    "h_320":    (HPLUS, "320", ["--aug", "shift", "--freeze-blocks", "24",
                                "--batch-p", "6"]),
    "h_parts":  (HPLUS, "256", ["--aug", "shift", "--freeze-blocks", "24",
                                "--batch-p", "8", "--parts"]),
    # --- этап 2: участники ансамбля, переобученные новым рецептом.
    # Это одновременно и проверка переноса рецепта, и holdout-модели для
    # честного замера итогового состава.
    "d_shift":  (DINOV3L, "256", ["--aug", "shift"]),
    # быстрый кандидат на роль основной модели: в паре с ViT-B·части даёт
    # 103 FPS против 51 у ViT-H+ (полный цикл по протоколу стенда)
    "d_ps":     (DINOV3L, "256", ["--aug", "shift", "--parts"]),
    # дистилляция структуры сходств от точного, но медленного ViT-H+@320 +
    # ViT-B·части (mAP@10 0.8828) в быструю DINOv3-L·части (0.8318 соло)
    # Вес 1.0 не дал ничего: KD-лосс ~0.003 против CE ~20 и triplet ~0.7
    # (траектория совпала с d_ps до четвёртого знака). Нужны сотни.
    "d_ps_kd100": (DINOV3L, "256", ["--aug", "shift", "--parts",
                                    "--kd", "runs/teacher_holdout.npz", "--kd-weight", "100"]),
    "d_ps_kd300": (DINOV3L, "256", ["--aug", "shift", "--parts",
                                    "--kd", "runs/teacher_holdout.npz", "--kd-weight", "300"]),
    "d_ps_kd1000": (DINOV3L, "256", ["--aug", "shift", "--parts",
                                     "--kd", "runs/teacher_holdout.npz", "--kd-weight", "1000"]),
    # --- надстройки над продовым рецептом (DINOv3-L·части + KD×1000) ---
    "k_xbm":   (DINOV3L, "256", ["--aug", "shift", "--parts", "--kd", "runs/teacher_holdout.npz",
                                 "--kd-weight", "1000", "--xbm", "4096"]),
    # XBM дал +0.0022 на одном сиде — столько же даёт смена сида без XBM,
    # поэтому нужен второй сид XBM для честного сравнения средних
    "k_xbm_s2": (DINOV3L, "256", ["--aug", "shift", "--parts", "--kd", "runs/teacher_holdout.npz",
                                  "--kd-weight", "1000", "--xbm", "4096", "--train-seed", "2"]),
    # --- поверх v4 (XBM): дольше, крупнее вход, сильнее дистилляция ---
    "k_kd3000": (DINOV3L, "256", ["--aug", "shift", "--parts", "--kd", "runs/teacher_holdout.npz",
                                  "--kd-weight", "3000", "--xbm", "4096"]),
    "k_288":    (DINOV3L, "288", ["--aug", "shift", "--parts", "--kd", "runs/teacher_holdout.npz",
                                  "--kd-weight", "1000", "--xbm", "4096"]),
    "k_ep50":   (DINOV3L, "256", ["--aug", "shift", "--parts", "--kd", "runs/teacher_holdout.npz",
                                  "--kd-weight", "1000", "--xbm", "4096", "--epochs", "50"]),
    "k_ema":   (DINOV3L, "256", ["--aug", "shift", "--parts", "--kd", "runs/teacher_holdout.npz",
                                 "--kd-weight", "1000", "--ema", "0.999"]),
    # второй и третий сид того же рецепта — для супа весов (src/soup.py)
    "k_s2":    (DINOV3L, "256", ["--aug", "shift", "--parts", "--kd", "runs/teacher_holdout.npz",
                                 "--kd-weight", "1000", "--train-seed", "2"]),
    "k_s3":    (DINOV3L, "256", ["--aug", "shift", "--parts", "--kd", "runs/teacher_holdout.npz",
                                 "--kd-weight", "1000", "--train-seed", "3"]),
    "l_shift":  ("vit_large_patch14_reg4_dinov2.lvd142m", "336",
                 ["--aug", "shift", "--batch-p", "8"]),
    "bp_shift": (VITB, "252", ["--aug", "shift", "--parts"]),
    "d_base":  (DINOV3L, "256", []),
    "d_best":  (DINOV3L, "320", ["--aug", "shift"]),
    # части-признаки: ViT-B от них вырос 0.751 -> 0.777 соло, но на крупные
    # модели их ни разу не применяли — самый очевидный незакрытый резерв
    "d_parts": (DINOV3L, "256", ["--parts"]),
    "l_parts": ("vit_large_patch14_reg4_dinov2.lvd142m", "336", ["--parts"]),
    "cnxl_v3": ("convnext_large.dinov3_lvd1689m", "256", ["--aug", "shift"]),
}


def _unpack(spec):
    """(backbone, размер, флаги[, каталог кропов]) — каталог необязателен."""
    backbone, size, extra = spec[0], spec[1], spec[2]
    return backbone, size, extra, (spec[3] if len(spec) > 3 else None)


def train_cmd(name, backbone, size, extra):
    size_args = [] if any(f == "--img-size" for f in extra) else ["--img-size", size]
    # gradient checkpointing замедляет примерно на треть, поэтому только там,
    # где без него не хватает 16 ГБ видеопамяти
    heavy = ["--grad-ckpt"] if ("large" in backbone or "huge" in backbone
                                or "giant" in backbone) else []
    return [sys.executable, "-m", "src.train", "--holdout-val",
            "--run-name", f"holdout_{name}", "--backbone", backbone,
            *size_args, *heavy, *extra]


def eval_cmd(name, backbone, size, extra=()):
    lb = ["--letterbox"] if "--letterbox" in extra else []
    return [sys.executable, "-m", "src.eval_candidate", "--tag", name,
            "--ckpt", str(RUNS / f"holdout_{name}" / "best.pt"),
            "--backbone", backbone, "--img-size", size, *lb]


def run(name, log_dir=RUNS / "logs"):
    backbone, size, extra, crops = _unpack(EXPERIMENTS[name])
    env = dict(os.environ)
    if crops:
        env["CROPS_DIR"] = crops
        print(f"[{name}] кропы из {crops}", flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt = RUNS / f"holdout_{name}" / "best.pt"
    if not ckpt.exists():
        print(f"[{name}] обучение...", flush=True)
        t0 = time.time()
        with open(log_dir / f"{name}.train.log", "w", encoding="utf-8") as f:
            r = subprocess.run(train_cmd(name, backbone, size, extra),
                               stdout=f, stderr=subprocess.STDOUT, env=env)
        if r.returncode != 0 or not ckpt.exists():
            print(f"[{name}] ОБУЧЕНИЕ УПАЛО (см. {log_dir / (name + '.train.log')})",
                  flush=True)
            return False
        print(f"[{name}] обучено за {(time.time()-t0)/60:.0f} мин", flush=True)
    else:
        print(f"[{name}] чекпойнт уже есть", flush=True)
    print(f"[{name}] оценка...", flush=True)
    with open(log_dir / f"{name}.eval.log", "w", encoding="utf-8") as f:
        r = subprocess.run(eval_cmd(name, backbone, size, extra),
                           stdout=f, stderr=subprocess.STDOUT, env=env)
    txt = (log_dir / f"{name}.eval.log").read_text(encoding="utf-8")
    for line in txt.splitlines():
        if "лучшее с кандидатом" in line or "соло " in line:
            print("   " + line.strip(), flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", nargs="*", default=None)
    ap.add_argument("--run-all", action="store_true")
    a = ap.parse_args()

    if a.list:
        for n, spec in EXPERIMENTS.items():
            bb, size, extra, crops = _unpack(spec)
            done = (RUNS / f"holdout_{n}" / "best.pt").exists()
            print(f"  {'[готово]' if done else '[  ждёт]'} {n:12s} {bb} @{size} "
                  f"{' '.join(extra)}{' crops=' + crops if crops else ''}")
        return
    names = list(EXPERIMENTS) if a.run_all else (a.run or [])
    for n in names:
        if n not in EXPERIMENTS:
            print(f"нет такого эксперимента: {n}")
            continue
        run(n)


if __name__ == "__main__":
    main()
