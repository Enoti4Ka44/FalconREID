"""Модель Re-ID: timm-backbone + BNNeck + классификационная голова с косинусным отступом.

Схема «сильного бейзлайна» ReID (Luo et al., 2019) + CosFace-отступ для
open-set устойчивости: на инференсе используется ТОЛЬКО эмбеддинг после
BNNeck, классификатор нужен лишь на обучении.
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginClassifier(nn.Module):
    """CosFace: s * (cos(theta) - m) для целевого класса."""

    def __init__(self, in_dim: int, num_classes: int, scale: float, margin: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scale, self.margin = scale, margin

    def forward(self, feat, labels=None):
        cos = F.linear(F.normalize(feat), F.normalize(self.weight))
        if labels is None:
            return cos * self.scale
        onehot = F.one_hot(labels, cos.size(1)).to(cos.dtype)
        return (cos - onehot * self.margin) * self.scale


class ReIDModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int, embed_dim: int = 768,
                 pretrained: bool = True, scale: float = 48.0, margin: float = 0.25,
                 img_size: int = 336):
        super().__init__()
        kwargs = dict(pretrained=pretrained, num_classes=0)  # без головы, глобальный пулинг timm
        if "vit" in backbone or "dinov2" in backbone:
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(backbone, **kwargs)
        feat_dim = self.backbone.num_features
        self.proj = nn.Identity() if feat_dim == embed_dim else nn.Linear(feat_dim, embed_dim, bias=False)
        # BNNeck: BN без смещения между triplet-пространством и классификатором
        self.bnneck = nn.BatchNorm1d(embed_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = MarginClassifier(embed_dim, num_classes, scale, margin) if num_classes > 0 else None

    def forward(self, x, labels=None):
        feat = self.proj(self.backbone(x))       # признак ДО BN — для triplet-лосса
        feat_bn = self.bnneck(feat)              # признак ПОСЛЕ BN — для классификатора и инференса
        if self.training and self.classifier is not None:
            logits = self.classifier(feat_bn, labels)
            return feat, feat_bn, logits
        return F.normalize(feat_bn, dim=1)       # инференс: L2-нормированный эмбеддинг

    @torch.no_grad()
    def extract(self, x, flip_tta: bool = True):
        """Эмбеддинг с TTA горизонтальным отражением (усреднение до нормировки)."""
        f = self.proj(self.backbone(x))
        f = self.bnneck(f)
        if flip_tta:
            f2 = self.proj(self.backbone(torch.flip(x, dims=[3])))
            f = f + self.bnneck(f2)
        return F.normalize(f, dim=1)


class ReIDModelParts(ReIDModel):
    """ReID с части-признаками (PCB-стиль поверх ViT-токенов).

    Глобальный путь идентичен ReIDModel; дополнительно patch-токены
    режутся на K горизонтальных полос (перед/центр/зад кузова в типичном
    ракурсе), каждая полоса — свой 256-d эмбеддинг с BNNeck и CosFace.
    Инференс: concat(глобальный 768, w_p * части K x 256), L2-норм.
    """

    K = 3
    PART_DIM = 256
    PART_W = 0.5

    def __init__(self, backbone, num_classes, embed_dim=768, pretrained=True,
                 scale=48.0, margin=0.25, img_size=252):
        super().__init__(backbone, num_classes, embed_dim, pretrained, scale, margin, img_size)
        self.part_proj = nn.ModuleList(
            nn.Linear(self.backbone.num_features, self.PART_DIM, bias=False)
            for _ in range(self.K))
        self.part_bn = nn.ModuleList(nn.BatchNorm1d(self.PART_DIM) for _ in range(self.K))
        for bn in self.part_bn:
            bn.bias.requires_grad_(False)
        self.part_cls = nn.ModuleList(
            MarginClassifier(self.PART_DIM, num_classes, scale, margin)
            for _ in range(self.K)) if num_classes > 0 else None

    def _tokens(self, x):
        f = self.backbone.forward_features(x)            # (B, prefix+N, C)
        npf = self.backbone.num_prefix_tokens
        patch = f[:, npf:]                               # (B, N, C)
        g = int(patch.shape[1] ** 0.5)
        return f, patch.reshape(patch.shape[0], g, g, -1)

    def _parts(self, grid):
        g = grid.shape[1]
        step = g // self.K
        return [grid[:, i * step:(i + 1) * step if i < self.K - 1 else g].mean(dim=(1, 2))
                for i in range(self.K)]

    def forward(self, x, labels=None):
        f, grid = self._tokens(x)
        pooled = self.backbone.forward_head(f, pre_logits=True)
        feat = self.proj(pooled)
        feat_bn = self.bnneck(feat)
        parts = self._parts(grid)
        parts_bn = [bn(pj(p)) for pj, bn, p in zip(self.part_proj, self.part_bn, parts)]
        if self.training and self.classifier is not None:
            logits = [self.classifier(feat_bn, labels)]
            logits += [cls(pb, labels) for cls, pb in zip(self.part_cls, parts_bn)]
            return feat, feat_bn, logits
        return self._concat(feat_bn, parts_bn)

    def _concat(self, feat_bn, parts_bn):
        out = [F.normalize(feat_bn, dim=1)]
        out += [self.PART_W * F.normalize(p, dim=1) for p in parts_bn]
        return F.normalize(torch.cat(out, dim=1), dim=1)

    @torch.no_grad()
    def extract(self, x, flip_tta: bool = True):
        def _one(inp):
            f, grid = self._tokens(inp)
            pooled = self.backbone.forward_head(f, pre_logits=True)
            fb = self.bnneck(self.proj(pooled))
            pb = [bn(pj(p)) for pj, bn, p in zip(self.part_proj, self.part_bn, self._parts(grid))]
            return fb, pb
        fb, pb = _one(x)
        if flip_tta:
            fb2, pb2 = _one(torch.flip(x, dims=[3]))
            fb = fb + fb2
            # при отражении части перед/зад не меняются местами (полосы горизонтальные)
            pb = [a + b for a, b in zip(pb, pb2)]
        return self._concat(fb, pb)
