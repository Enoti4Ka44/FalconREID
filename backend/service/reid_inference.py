"""ReIDInference — загрузка моделей ансамбля и инференс.

Использует torch. Не знает про галерею.
"""
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ReIDModel, ReIDModelParts  # noqa: E402

IMG_SIZE = 252
IMG_SIZE2 = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BBOX_PAD = 0.06


def _load_one(path, img_size, device):
    from src.int8_pack import load_packed
    ck = load_packed(Path(path))
    backbone = str(ck.get("cfg", {}).get("backbone", "vit_base_patch14_reg4_dinov2.lvd142m"))
    kwargs = {"img_size": img_size} if ("vit" in backbone or "dinov2" in backbone) else {}
    cls = ReIDModelParts if any(k.startswith("part_proj.") for k in ck["model"]) else ReIDModel
    m = cls(backbone, ck["num_classes"], 768, pretrained=False, **kwargs)
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def _has_plain_attention(model) -> bool:
    """ViT из timm с обычным Attention (qkv + scale, без RoPE)."""
    bb = getattr(model, "backbone", None)
    blocks = getattr(bb, "blocks", None)
    if blocks is None or getattr(bb, "rope", None) is not None:
        return False
    attn = blocks[-1].attn
    return hasattr(attn, "qkv") and hasattr(attn, "scale") and not hasattr(attn, "q_bias")


def _make_tf(size):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


class ReIDInference:
    """Ансамбль моделей для извлечения эмбеддингов и attention-карт."""

    def __init__(self, model_path: str, gallery_index: Path,
                 model2_path: str | None = None, w2: float = 0.5):
        import json as _json
        import os

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        manifest = Path(gallery_index).parent / "ensemble.json"
        self.members = []

        if manifest.exists():
            spec = _json.loads(manifest.read_text(encoding="utf-8"))["models"]
            base = Path(gallery_index).parent
            for m in spec:
                ckpt = Path(m["ckpt"])
                if not ckpt.exists():
                    ckpt = base / Path(m["ckpt"]).name
                if not ckpt.exists():
                    print(f"[inference] вес не найден, модель пропущена: {m['ckpt']}")
                    continue
                self.members.append({
                    "model": _load_one(str(ckpt), m["img_size"], self.device),
                    "tf": _make_tf(m["img_size"]),
                    "weight": float(m["weight"]),
                })

        if not self.members:
            self.members.append({
                "model": _load_one(model_path, IMG_SIZE, self.device),
                "tf": _make_tf(IMG_SIZE),
                "weight": 1.0,
            })
            model2_path = model2_path or os.environ.get("MODEL2_PATH")
            if model2_path and Path(model2_path).exists():
                self.members.append({
                    "model": _load_one(model2_path, IMG_SIZE2, self.device),
                    "tf": _make_tf(IMG_SIZE2),
                    "weight": w2,
                })

        # Карта внимания строится по участнику со стандартным ViT-вниманием
        # (DINOv2 ViT-B): у DINOv3 внимание с RoPE и раздельными смещениями q/v,
        # ручной пересчёт весов внимания для него некорректен.
        attn_member = next((m for m in self.members
                            if _has_plain_attention(m["model"])), self.members[0])
        self.model = attn_member["model"]
        self.tf = attn_member["tf"]
        self.n_models = len(self.members)

        # прогрев: первый проход на GPU инициализирует ядра cuDNN/cuBLAS
        # (~0.8 с) — пусть это случится при старте, а не на запросе оператора
        dummy = Image.new("RGB", (320, 240), (128, 128, 128))
        for _ in range(2):
            self.embed(dummy)

    @staticmethod
    def crop_bbox(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
        px, py = w * BBOX_PAD, h * BBOX_PAD
        x0 = max(0, int(x - px))
        y0 = max(0, int(y - py))
        x1 = min(img.width, int(x + w + px))
        y1 = min(img.height, int(y + h + py))
        return img.crop((x0, y0, x1, y1))

    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray:
        parts = []
        with torch.autocast("cuda", enabled=self.device == "cuda"):
            for mm in self.members:
                f = mm["model"].extract(mm["tf"](img).unsqueeze(0).to(self.device))
                parts.append(mm["weight"] * f.float().cpu().numpy()[0])
        f = np.concatenate(parts)
        return f / np.linalg.norm(f)

    @torch.no_grad()
    def attention_map(self, img: Image.Image) -> bytes:
        """Карта внимания последнего блока ViT (CLS -> патчи), наложенная на кроп."""
        import matplotlib
        matplotlib.use("Agg")

        x = self.tf(img).unsqueeze(0).to(self.device)
        bb = self.model.backbone
        attn_out = {}

        blk = bb.blocks[-1]

        def hook(module, inp, out):
            attn_out["x"] = inp[0]

        h = blk.attn.register_forward_hook(hook)
        bb(x)
        h.remove()

        t = attn_out["x"]          # вход attention уже прошёл norm1 внутри блока
        a = blk.attn
        B, N, C = t.shape
        qkv = a.qkv(t).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * a.scale
        attn = attn.softmax(dim=-1)
        n_prefix = getattr(bb, "num_prefix_tokens", 1)
        cls_attn = attn[0, :, 0, n_prefix:].mean(0)
        g = int(np.sqrt(cls_attn.numel()))
        amap = cls_attn[: g * g].reshape(g, g).float().cpu().numpy()
        # отдельные «высоконормные» токены ViT забирают почти всё внимание CLS;
        # обрезка по 99-му перцентилю не даёт им погасить остальную карту
        amap = np.clip(amap, None, np.percentile(amap, 99))
        amap = (amap - amap.min()) / (np.ptp(amap) + 1e-9)

        cmap = matplotlib.colormaps["inferno"]
        W = min(img.width, 640)
        base = img.resize((W, max(1, round(img.height * W / img.width))), Image.BILINEAR)
        amap_img = Image.fromarray((amap * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)
        heat = Image.fromarray(
            (cmap(np.array(amap_img) / 255.0)[:, :, :3] * 255).astype(np.uint8))
        out = Image.blend(base, heat, 0.45)
        buf = io.BytesIO()
        out.save(buf, "PNG")
        return buf.getvalue()
