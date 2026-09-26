"""Мультикадровый агрегатор: обучаемое attention-взвешивание кадров запроса.

Сейчас мульти-запрос усредняет эмбеддинги 1–5 снимков поровну. Агрегатор
учит маленький MLP оценивать «качество» кадра по его эмбеддингу и
взвешивать среднее: смазанные/ночные кадры получают меньший вес.

Честная постановка (без утечки):
  * обучение — эмбеддинги holdout-квада на train-идентичностях
    (runs/distill/teacher_train_holdout.npy, считается src.distill);
  * оценка — замороженный эталон: запросы val_q группируются по
    (машина, камера) = мульти-запрос оператора, галерея val_g,
    кросс-камерный протокол; сравнение attention против простого среднего.

Запуск: python -m src.distill (один раз, для эмбеддингов) -> python -m src.aggregator
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .dataset import build_val_protocol, split_train_val

TRAIN_EMB = Path("runs/distill/teacher_train_holdout.npy")
VAL_PARTS = [("vitl", 1.0), ("vit", 0.7), ("m2", 0.5), ("dinov3", 1.0)]
CKPT = Path("runs/aggregator.pt")   # не входит в прод-ансамбль: среднее не хуже


class FrameQuality(nn.Module):
    """Скалярный вес кадра по его эмбеддингу (софтмакс внутри группы)."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 4), nn.GELU(),
            nn.Linear(hidden // 4, 1))

    def forward(self, embs, mask):
        w = self.net(embs).squeeze(-1)
        w = w.masked_fill(~mask, -1e4).softmax(dim=1)
        agg = (embs * w.unsqueeze(-1)).sum(1)
        return F.normalize(agg, dim=-1), w


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def groups_of(vids, cams, min_size=2):
    g = defaultdict(list)
    for i, (v, c) in enumerate(zip(vids, cams)):
        g[(v, c)].append(i)
    return {k: v for k, v in g.items() if len(v) >= min_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--tmax", type=int, default=5)
    a = ap.parse_args()
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed)

    df = pd.read_csv(cfg.data_root / "train.csv")
    train_df, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
    train_df = train_df.reset_index(drop=True)
    emb = torch.tensor(l2(np.load(TRAIN_EMB)), dtype=torch.float32, device=device)
    vids, cams = train_df.vehicle_id.values, train_df.camera_id.values
    groups = groups_of(vids, cams)
    by_vid = defaultdict(list)
    for i, v in enumerate(vids):
        by_vid[v].append(i)
    keys = [k for k in groups if any(cams[i] != k[1] for i in by_vid[k[0]])]
    print(f"train: групп мульти-запроса {len(keys)}, dim={emb.shape[1]}")

    model = FrameQuality(emb.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    D = emb.shape[1]
    for ep in range(a.epochs):
        rng.shuffle(keys)
        tot = n = 0
        for b0 in range(0, len(keys), 64):
            E, M, T = [], [], []
            for (v, c) in keys[b0:b0 + 64]:
                idx = groups[(v, c)]
                take = rng.choice(idx, size=min(len(idx), a.tmax), replace=False)
                tgt = int(rng.choice([i for i in by_vid[v] if cams[i] != c]))
                pad = torch.zeros(a.tmax - len(take), D, device=device)
                E.append(torch.cat([emb[take], pad]))
                M.append(torch.tensor([True] * len(take) + [False] * (a.tmax - len(take)),
                                      device=device))
                T.append(tgt)
            agg, _ = model(torch.stack(E), torch.stack(M))
            logits = agg @ emb[T].T / 0.05
            loss = F.cross_entropy(logits, torch.arange(len(agg), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(agg); n += len(agg)
        if (ep + 1) % 10 == 0:
            print(f"ep {ep + 1}/{a.epochs} loss={tot / n:.4f}")

    # ---------- честная оценка на замороженном эталоне ----------
    vq, vg = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
    parts_q, parts_g = [], []
    for tag, w in VAL_PARTS:
        d = np.load(f"artifacts/val_embeddings_{tag}.npz")
        parts_q.append(w * d["q"]); parts_g.append(w * d["g"])
    Q = torch.tensor(l2(np.concatenate(parts_q, 1)), device=device)
    G = torch.tensor(l2(np.concatenate(parts_g, 1)), device=device)
    q_vid, q_cam = vq.vehicle_id.values, vq.camera_id.values
    g_vid, g_cam = vg.vehicle_id.values, vg.camera_id.values

    model.eval()
    res = {"single": [0, 0], "mean": [0, 0], "attention": [0, 0]}
    with torch.no_grad():
        for (v, c), idx in groups_of(q_vid, q_cam).items():
            valid = (g_vid != v) | (g_cam != c)             # свою камеру исключаем
            if not ((g_vid == v) & (g_cam != c)).any():
                continue                                    # у группы нет пары — пропуск
            take = idx[: a.tmax]
            vecs = {
                "single": Q[take[:1]],
                "mean": F.normalize(Q[take].mean(0, keepdim=True), dim=-1),
                "attention": model(Q[take].unsqueeze(0),
                                   torch.ones(1, len(take), dtype=torch.bool, device=device))[0],
            }
            for k, vec in vecs.items():
                s = (vec @ G.T).squeeze(0)
                s[~torch.tensor(valid, device=device)] = -2
                res[k][0] += int(g_vid[int(s.argmax())] == v)
                res[k][1] += 1
    out = {k: round(h / max(t, 1), 4) for k, (h, t) in res.items()}
    out["groups"] = res["mean"][1]
    print(f"эталон, мульти-запрос R1: один кадр={out['single']}  "
          f"среднее={out['mean']}  attention={out['attention']}  (групп {out['groups']})")
    torch.save({"model": model.state_dict(), "dim": D, "eval": out}, CKPT)
    Path("runs/aggregator_eval.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
