"""Графика для презентации и документации.

Вход: artifacts/val_curves.json, artifacts/error_cases.json, runs/*/history.json
Выход: artifacts/figures/*.png (палитра шаблона ЛЦТ-2026, тёмный фон #1C1D22)
"""
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# палитра шаблона ЛЦТ
BG = "#1C1D22"
FG = "#FFFFFF"
MUT = "#9A93C9"
ACC = "#FF0053"      # основной акцент
ACC2 = "#8A83D1"     # сиреневый
ACC3 = "#FFD6E4"     # розовый светлый
GRID = "#33343C"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": GRID, "axes.labelcolor": FG,
    "xtick.color": MUT, "ytick.color": MUT, "font.size": 12,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "Segoe UI",
})

ART = Path("artifacts")
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def fig_threshold_curves():
    c = json.loads((ART / "val_curves.json").read_text())
    thr_info = json.loads(Path("models/threshold.json").read_text())
    t = np.array(c["grid"])
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    ax.plot(t, c["F1"], color=ACC, lw=2.5, label="F1")
    ax.plot(t, c["TNR"], color=ACC2, lw=2.5, label="TNR (верные отказы)")
    ax.plot(t, c["recall"], color=ACC3, lw=1.6, ls="--", label="Recall")
    tau = thr_info["threshold"]
    ax.axvline(tau, color=FG, lw=1.2, ls=":", alpha=.8)
    ax.annotate(f"τ = {tau:.3f}", (tau, 0.06), xytext=(tau + 0.03, 0.06),
                color=FG, fontsize=12, fontweight="bold")
    ax.set_xlabel("Порог косинусной близости")
    ax.set_ylabel("Значение метрики")
    ax.set_xlim(t.min(), min(1.0, t.max()))
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", facecolor=BG, edgecolor=GRID, framealpha=.9)
    fig.tight_layout()
    fig.savefig(FIG / "threshold_curves.png")
    plt.close(fig)


def fig_similarity_hist():
    c = json.loads((ART / "val_curves.json").read_text())
    tau = json.loads(Path("models/threshold.json").read_text())["threshold"]
    m, nm = np.array(c["max_sim_match"]), np.array(c["max_sim_nomatch"])
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    bins = np.linspace(min(nm.min(), m.min()), 1.0, 55)
    ax.hist(nm, bins=bins, color=ACC2, alpha=.75, label="Пары в галерее НЕТ (дистракторы)")
    ax.hist(m, bins=bins, color=ACC, alpha=.75, label="Пара в галерее ЕСТЬ")
    ax.axvline(tau, color=FG, lw=1.4, ls=":")
    ax.annotate(f"порог τ = {tau:.3f}\nслева — отказ", (tau, ax.get_ylim()[1] * .82),
                xytext=(tau + .02, ax.get_ylim()[1] * .8), color=FG, fontsize=11)
    ax.set_xlabel("Максимальная близость запроса к галерее")
    ax.set_ylabel("Число запросов")
    ax.legend(facecolor=BG, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(FIG / "similarity_hist.png")
    plt.close(fig)


def fig_training_curve(run="runs/holdout/history.json"):
    hist = json.loads(Path(run).read_text())
    eps = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.plot(eps, [h["ce"] for h in hist], color=ACC2, lw=2, label="CE (CosFace)")
    ax.plot(eps, [h["tri"] for h in hist], color=ACC3, lw=2, label="Triplet")
    vals = [(h["epoch"], h["mAP"]) for h in hist if "mAP" in h]
    if vals:
        ax2 = ax.twinx()
        ax2.plot(*zip(*vals), color=ACC, lw=2.5, marker="o", label="val mAP")
        ax2.set_ylabel("mAP", color=ACC)
        ax2.tick_params(axis="y", colors=ACC)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(GRID)
        ax2.grid(False)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Функция потерь")
    ax.legend(loc="upper right", facecolor=BG, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(FIG / "training_curve.png")
    plt.close(fig)


def strip(pairs, out, crops=Path(os.environ.get("CROPS_DIR", "data/crops")), h=220, label_pairs=True):
    """Полоса примеров: запрос | топ-1 (для слайда анализа ошибок)."""
    import os
    crops = Path(os.environ.get("DATA_ROOT", crops.parent)) / "crops"
    cells = []
    for p in pairs:
        for key in ("query", "top1"):
            im = Image.open(crops / f"{p[key]}.jpg").convert("RGB")
            s = h / im.height
            cells.append(im.resize((int(im.width * s), h)))
    w = sum(c.width for c in cells) + 8 * (len(cells) - 1)
    canvas = Image.new("RGB", (w, h), BG)
    x = 0
    for c in cells:
        canvas.paste(c, (x, 0))
        x += c.width + 8
    canvas.save(FIG / out)


def fig_error_strips():
    e = json.loads((ART / "error_cases.json").read_text())
    if e["rank1_miss"]:
        strip(e["rank1_miss"][:2], "errors_rank1miss.png", h=300)
    if e["false_accept_distractor"]:
        strip(e["false_accept_distractor"][:2], "errors_false_accept.png", h=300)
    if e["hard_correct"]:
        strip(e["hard_correct"][:2], "hard_correct.png", h=300)


if __name__ == "__main__":
    fig_threshold_curves()
    fig_similarity_hist()
    try:
        fig_training_curve()
    except FileNotFoundError:
        print("history.json не найден — пропускаю кривую обучения")
    fig_error_strips()
    print("figures ->", FIG)
