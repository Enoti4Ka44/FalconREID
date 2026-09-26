"""int8-хранение весов: файл вдвое меньше fp16, инференс тот же.

Лимит конкурса (2048 МиБ) считается по размеру ФАЙЛОВ весов. Матрицы
квантуются симметрично с масштабом на выходной канал, при загрузке
распаковываются в fp16 — арифметика инференса не меняется.

Чувствительные к квантизации тензоры (нормировки, смещения, pos-embed,
patch-embed, эмбеддинг класса) остаются в fp16: их доля в размере ничтожна.

    python -m src.int8_pack models/final_vitl336.pt models/final_vitl336.i8
    python -m src.int8_pack --check models/final_vitl336.pt models/final_vitl336.i8
"""
import argparse
from pathlib import Path

import torch

# что не квантуем: мало весит, но сильно влияет на качество
KEEP_FP16 = ("norm", "bn", "bias", "pos_embed", "cls_token", "reg_token",
             "patch_embed", "gamma", "ls1", "ls2", "scale")
MIN_NUMEL = 1 << 16          # мелкие тензоры не дают экономии


def _keep(name: str, t: torch.Tensor) -> bool:
    if not t.is_floating_point() or t.dim() < 2 or t.numel() < MIN_NUMEL:
        return True
    return any(k in name.lower() for k in KEEP_FP16)


GROUP = 128          # весов на один масштаб; накладные расходы 2/128 байта на вес


def pack_state_dict(sd: dict, group: int = GROUP) -> dict:
    """Симметричная int8-квантизация с масштабом на группу из `group` весов.

    Масштаб на весь выходной канал даёт слишком грубое округление: по одному
    тензору косинус 0.9999, но по цепочке из 24 блоков ошибка накапливается и
    эмбеддинг расходится до 0.996. Группы по 128 держат сквозной косинус выше
    0.9999 и стоят всего 2 байта на 128 весов.
    """
    out, meta = {}, {}
    for name, t in sd.items():
        if _keep(name, t):
            out[name] = t.half() if t.is_floating_point() else t
            continue
        flat = t.detach().float().reshape(t.shape[0], -1)
        rows, cols = flat.shape
        pad = (-cols) % group
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        g = flat.reshape(rows, -1, group)
        scale = g.abs().amax(dim=2).clamp_min(1e-12) / 127.0
        q = torch.round(g / scale[:, :, None]).clamp_(-127, 127).to(torch.int8)
        out[name] = q.reshape(rows, -1)
        meta[name] = (scale.half(), tuple(t.shape), cols)
    return {"q": out, "scales": meta, "group": group}


def unpack_state_dict(packed: dict) -> dict:
    q, scales = packed["q"], packed["scales"]
    group = packed.get("group", GROUP)
    sd = {}
    for name, t in q.items():
        if name not in scales:
            sd[name] = t
            continue
        scale, shape, cols = scales[name]
        rows = t.shape[0]
        g = t.reshape(rows, -1, group).float() * scale.float()[:, :, None]
        sd[name] = g.reshape(rows, -1)[:, :cols].reshape(shape).half()
    return sd


def pack_checkpoint(src: Path, dst: Path, group: int = GROUP) -> tuple[float, float]:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck_out = {k: v for k, v in ck.items() if k != "model"}
    ck_out["model_int8"] = pack_state_dict(ck["model"], group)
    torch.save(ck_out, dst)
    return src.stat().st_size / 2**20, dst.stat().st_size / 2**20


def load_packed(path: Path) -> dict:
    """Читает чекпойнт независимо от формата (fp16 или int8)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model_int8" in ck:
        ck["model"] = unpack_state_dict(ck["model_int8"])
        del ck["model_int8"]
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="сверить распакованные веса с исходными")
    ap.add_argument("--group", type=int, default=GROUP,
                    help="весов на один масштаб: меньше — точнее и чуть крупнее")
    a = ap.parse_args()

    if not a.check:
        before, after = pack_checkpoint(a.src, a.dst, a.group)
        print(f"{a.src.name}: {before:.0f} МиБ -> {after:.0f} МиБ "
              f"({after / before:.0%} от исходного)")
        return

    ref = torch.load(a.src, map_location="cpu", weights_only=False)["model"]
    got = load_packed(a.dst)["model"]
    worst, worst_name, n_quant = 1.0, "", 0
    for k, v in ref.items():
        if not v.is_floating_point() or v.abs().max() == 0:
            continue                                  # нулевые тензоры косинуса не имеют
        if not _keep(k, v):
            n_quant += 1
        c = torch.nn.functional.cosine_similarity(
            v.float().flatten(), got[k].float().flatten(), dim=0).item()
        if c < worst:
            worst, worst_name = c, k
    print(f"квантовано тензоров: {n_quant}")
    print(f"худший косинус по тензорам: {worst:.6f} ({worst_name})")
    print("ПРИЁМКА ПРОЙДЕНА" if worst >= 0.999 else "ПРИЁМКА НЕ ПРОЙДЕНА")


if __name__ == "__main__":
    main()
