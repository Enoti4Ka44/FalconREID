"""Обучение Re-ID модели.

Запуск:  python -m src.train [--epochs 30] [--val]  (из каталога solution)
Рецепт: DINOv2 ViT-B/14 + BNNeck, CosFace + batch-hard triplet,
PK-сэмплинг, косинусный LR c warmup, AMP.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .dataset import CropDataset, PKSampler, build_transforms, split_train_val, build_val_protocol
from .eval_utils import best_refusal_threshold, compute_reid_metrics, refusal_metrics
from .losses import CrossBatchMemory, batch_hard_triplet, similarity_kd, xbm_triplet
from .model import ReIDModel


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_embeddings(model, df, cfg, device, bs=64):
    tf = build_transforms(cfg.img_size, cfg.mean, cfg.std, train=False,
                          letterbox=getattr(cfg, "letterbox", False))
    ds = CropDataset(df, cfg.crops_dir, tf, with_labels=False)
    dl = DataLoader(ds, batch_size=bs, num_workers=cfg.num_workers, pin_memory=True)
    model.eval()
    out = []
    with torch.autocast("cuda", enabled=cfg.amp):
        for img, _, _ in dl:
            out.append(model.extract(img.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def validate(model, val_q, val_g, cfg, device):
    q_emb = extract_embeddings(model, val_q, cfg, device)
    g_emb = extract_embeddings(model, val_g, cfg, device)
    m = compute_reid_metrics(
        q_emb, g_emb,
        val_q["vehicle_id"].values, val_g["vehicle_id"].values,
        val_q["camera_id"].values, val_g["camera_id"].values,
    )
    # режим отказа: есть ли у запроса кросс-камерный позитив в галерее
    sim = q_emb @ g_emb.T
    same_cam = (val_q["camera_id"].values[:, None] == val_g["camera_id"].values[None, :])
    sim_x = np.where(same_cam, -1.0, sim)            # кросс-камерная близость
    max_sim = sim_x.max(axis=1)
    has_match = np.array([
        ((val_g["vehicle_id"].values == v) & (val_g["camera_id"].values != c)).any()
        for v, c in zip(val_q["vehicle_id"], val_q["camera_id"])
    ])
    thr = best_refusal_threshold(max_sim, has_match)
    m.update({"refusal_bestF1": thr["F1"], "refusal_TNR": thr["TNR"],
              "refusal_threshold": thr["threshold"]})
    return m


def _llrd_groups(model, base_lr, decay):
    """Группы параметров backbone с LR, растущим от нижних блоков к верхним."""
    import re
    blocks = {}
    for n, p in model.named_parameters():
        if not n.startswith("backbone") or not p.requires_grad:
            continue
        m = re.search(r"\.(?:blocks|layers|stages)\.(\d+)\.", n)
        blocks.setdefault(int(m.group(1)) if m else -1, []).append(p)
    depth = max(blocks) + 1 if blocks else 1
    return [{"params": ps, "lr": base_lr * decay ** (depth - 1 - max(i, 0))}
            for i, ps in sorted(blocks.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--holdout-val", action="store_true",
                    help="отложить 20%% идентичностей на валидацию (для подбора порога); "
                         "без флага обучение идёт на всех данных")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--batch-p", type=int, default=None,
                    help="идентичностей в батче (PK-сэмплер)")
    ap.add_argument("--batch-k", type=int, default=None,
                    help="снимков на идентичность")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing (для крупных backbone на 12ГБ GPU)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--extra-data", default=None,
                    help="parquet внешних данных (src.external_data) — добавляется к train")
    ap.add_argument("--extra-only", action="store_true",
                    help="этап предобучения: учиться ТОЛЬКО на внешних данных")
    ap.add_argument("--init-from", default=None,
                    help="чекпойнт для инициализации backbone (двухэтапное обучение)")
    ap.add_argument("--parts", action="store_true",
                    help="части-признаки: PCB-полосы поверх ViT-токенов")
    ap.add_argument("--aug", default="base",
                    help="геометрические аугментации: base | shift[N] | rrc "
                         "(N — доля сдвига в процентах, по умолчанию 5)")
    ap.add_argument("--img-h", type=int, default=None,
                    help="высота входа, если нужен НЕквадратный вход (ширина — --img-size)")
    ap.add_argument("--letterbox", action="store_true",
                    help="вписывать кроп с сохранением пропорций вместо растягивания")
    ap.add_argument("--cam-aware", action="store_true",
                    help="PK-сэмплер берёт K снимков ТС с РАЗНЫХ камер")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="коэффициент EMA весов (напр. 0.999); 0 — выключено")
    ap.add_argument("--llrd", type=float, default=0.0,
                    help="послойное затухание LR (напр. 0.75); 0 — выключено")
    ap.add_argument("--lr", type=float, default=None,
                    help="LR головы; при замороженном backbone нужен заметно "
                         "больший, чем стандартные 1e-4")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="сид ОБУЧЕНИЯ (инициализация, порядок батчей) — для супа весов. "
                         "Разбиение train/val всегда на cfg.seed, иначе замер нечестен")
    ap.add_argument("--kd", default=None,
                    help="npz с эмбеддингами учителя (src.teacher_emb) для дистилляции")
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--kd-temp", type=float, default=0.1)
    ap.add_argument("--xbm", type=int, default=0,
                    help="размер памяти признаков из прошлых батчей (напр. 2048); "
                         "даёт трудные негативы, когда батч маленький")
    ap.add_argument("--xbm-start", type=int, default=2,
                    help="с какой эпохи включать память (после прогрева)")
    ap.add_argument("--freeze-blocks", type=int, default=0,
                    help="заморозить первые N блоков backbone; -1 — весь backbone "
                         "(нужно для моделей, которые целиком не влезают в 16 ГБ)")
    a = ap.parse_args()

    cfg = Config()
    if a.epochs: cfg.epochs = a.epochs
    if a.backbone: cfg.backbone = a.backbone
    if a.img_size: cfg.img_size = a.img_size
    if a.img_h: cfg.img_size = (a.img_h, a.img_size)
    if a.batch_p: cfg.batch_p = a.batch_p
    if a.batch_k: cfg.batch_k = a.batch_k
    if a.lr: cfg.lr = a.lr
    if a.letterbox: cfg.letterbox = True
    train_seed = a.train_seed if a.train_seed is not None else cfg.seed
    seed_all(train_seed)
    device = "cuda"
    run_dir = cfg.work_dir / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.data_root / "train.csv")
    if a.holdout_val:
        train_df, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
        val_q, val_g = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
        print(f"train ids={train_df.vehicle_id.nunique()} imgs={len(train_df)} | "
              f"val q={len(val_q)} g={len(val_g)}")
    else:
        train_df, val_q, val_g = df, None, None
        print(f"train on ALL: ids={df.vehicle_id.nunique()} imgs={len(df)}")

    if a.extra_data:
        ext = pd.read_parquet(a.extra_data)
        ext = ext.assign(image_id=ext.image_path)      # для единообразия колонок
        if a.extra_only:
            train_df = ext[["image_id", "image_path", "vehicle_id", "camera_id"]].copy()
            print(f"предобучение ТОЛЬКО на внешних: {len(train_df)} изображений, "
                  f"{train_df.vehicle_id.nunique()} идентичностей")
        else:
            # единый строковый тип ID, чтобы sorted() в vid2label не падал
            train_df = train_df.assign(vehicle_id=train_df.vehicle_id.astype(str),
                                       image_path=None)
            train_df = pd.concat(
                [train_df, ext[["image_id", "image_path", "vehicle_id", "camera_id"]]],
                ignore_index=True)
            print(f"+внешние данные: {len(ext)} изображений, "
                  f"{ext.vehicle_id.nunique()} идентичностей "
                  f"(итого ids={train_df.vehicle_id.nunique()}, imgs={len(train_df)})")

    tf_train = build_transforms(cfg.img_size, cfg.mean, cfg.std, train=True, aug=a.aug,
                                letterbox=cfg.letterbox)
    print(f"аугментации: {a.aug}, вход {cfg.img_size}")
    ds = CropDataset(train_df, cfg.crops_dir, tf_train, with_labels=True,
                     return_index=bool(a.kd))
    teacher = None
    if a.kd:
        kd = np.load(a.kd)
        pos = {iid: k for k, iid in enumerate(kd["image_id"])}
        miss = [i for i in ds.df.image_id if i not in pos]
        if miss:
            raise SystemExit(f"у учителя нет эмбеддингов для {len(miss)} снимков, "
                             f"например {miss[:3]}")
        teacher = torch.from_numpy(
            kd["emb"][[pos[i] for i in ds.df.image_id]].astype(np.float32)).to(device)
        print(f"дистилляция: учитель {tuple(teacher.shape)}, вес {a.kd_weight}, "
              f"температура {a.kd_temp}")
    sampler = PKSampler(
        [ds.vid2label[v] for v in train_df["vehicle_id"]], cfg.batch_p, cfg.batch_k,
        train_seed, cameras=train_df["camera_id"].values if a.cam_aware else None)
    if a.cam_aware:
        print("PK-сэмплер: камеро-осознанный")
    dl = DataLoader(ds, batch_size=cfg.batch_p * cfg.batch_k, sampler=sampler,
                    num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=True)

    from .model import ReIDModelParts
    model_cls = ReIDModelParts if a.parts else ReIDModel
    model = model_cls(cfg.backbone, ds.num_classes, cfg.embed_dim, cfg.pretrained,
                      cfg.arcface_scale, cfg.arcface_margin, cfg.img_size).to(device)
    if a.grad_ckpt and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)
        print("gradient checkpointing: ON")

    if a.freeze_blocks:
        import re as _re
        n_frozen = 0
        for n, prm in model.named_parameters():
            if not n.startswith("backbone"):
                continue
            mt = _re.search(r"\.(?:blocks|layers|stages)\.(\d+)\.", n)
            blk = int(mt.group(1)) if mt else -1
            if a.freeze_blocks < 0 or blk < a.freeze_blocks:
                prm.requires_grad_(False)
                n_frozen += prm.numel()
        total = sum(p.numel() for p in model.parameters())
        print(f"заморожено {n_frozen/1e6:.0f}М из {total/1e6:.0f}М параметров "
              f"({'весь backbone' if a.freeze_blocks < 0 else f'первые {a.freeze_blocks} блоков'})")

    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone") and p.requires_grad]
    bb_params = [p for n, p in model.named_parameters()
                 if n.startswith("backbone") and p.requires_grad]
    if a.llrd > 0 and bb_params:
        # Послойное затухание: нижние блоки (общие признаки) учатся медленнее
        # верхних. На малом датасете это заметно снижает переобучение.
        groups = [g for g in _llrd_groups(model, cfg.lr * cfg.backbone_lr_scale, a.llrd)
                  if g["params"]]
        opt = torch.optim.AdamW([{"params": head_params, "lr": cfg.lr}] + groups,
                                weight_decay=cfg.weight_decay)
        print(f"послойное затухание LR: {a.llrd} ({len(groups)} групп)")
    else:
        groups = [{"params": head_params, "lr": cfg.lr}]
        if bb_params:
            groups.append({"params": bb_params, "lr": cfg.lr * cfg.backbone_lr_scale})
        opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(dl)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda] * len(opt.param_groups))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    start_ep = 0
    if a.init_from:
        ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
        sd = ck["model"]
        # переносим backbone/BNNeck/эмбеддинг; классиф. голова другая (число классов)
        own = model.state_dict()
        keep = {k: v for k, v in sd.items()
                if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"init from {a.init_from}: перенесено {len(keep)}/{len(own)} тензоров")

    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep = ck["epoch"] + 1
        print(f"resumed from {a.resume} at epoch {start_ep}")

    # EMA: экспоненциальное среднее весов. Сглаживает шум малого датасета,
    # на инференсе не стоит ничего. Оцениваются ОБА набора, в best.pt уходит
    # тот, что лучше на отложенной валидации.
    ema_model = None
    if a.ema > 0:
        import copy as _copy
        ema_model = _copy.deepcopy(model).eval()
        for prm in ema_model.parameters():
            prm.requires_grad_(False)
        print(f"EMA весов: decay={a.ema}")

    def ema_update():
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in ema_model.state_dict().items():
                if v.is_floating_point():
                    v.mul_(a.ema).add_(msd[k].detach(), alpha=1 - a.ema)
                else:
                    v.copy_(msd[k])

    xbm = CrossBatchMemory(a.xbm, cfg.embed_dim, device) if a.xbm > 0 else None
    if xbm is not None:
        print(f"память по батчам (XBM): {a.xbm} признаков, с эпохи {a.xbm_start + 1}")

    best_map = 0.0
    history = []
    for ep in range(start_ep, cfg.epochs):
        model.train()
        t0, tot_ce, tot_tri, tot_acc, n = time.time(), 0.0, 0.0, 0.0, 0
        tot_kd = 0.0
        for batch in dl:
            img, label = batch[0], batch[1]
            idx = batch[3] if teacher is not None else None
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=cfg.amp):
                feat, feat_bn, logits = model(img, label)
                if isinstance(logits, list):     # части: CE усредняется по головам
                    ce = sum(F.cross_entropy(l, label, label_smoothing=cfg.label_smoothing)
                             for l in logits) / len(logits)
                    logits = logits[0]
                else:
                    ce = F.cross_entropy(logits, label, label_smoothing=cfg.label_smoothing)
                tri = batch_hard_triplet(feat.float(), label, cfg.triplet_margin)
                loss = ce + cfg.triplet_weight * tri
                if xbm is not None and ep >= a.xbm_start:
                    loss = loss + cfg.triplet_weight * xbm_triplet(
                        feat.float(), label, *xbm.get(), cfg.triplet_margin)
                if teacher is not None:
                    kd_val = similarity_kd(feat_bn, teacher[idx.to(device)], a.kd_temp)
                    loss = loss + a.kd_weight * kd_val
                    tot_kd += kd_val.item() * len(label)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # только после backward: память участвует в графе xbm_triplet,
            # запись в неё до обратного прохода ломает autograd (версия тензора)
            if xbm is not None:
                xbm.enqueue(feat.float(), label)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema_model is not None:
                ema_update()
            bs = len(label)
            tot_ce += ce.item() * bs; tot_tri += tri.item() * bs
            tot_acc += (logits.argmax(1) == label).float().sum().item(); n += bs
        kd_txt = f"kd={tot_kd / n:.4f} " if teacher is not None else ""
        line = (f"ep {ep + 1}/{cfg.epochs} ce={tot_ce / n:.3f} tri={tot_tri / n:.3f} {kd_txt}"
                f"acc={tot_acc / n:.3f} lr={sched.get_last_lr()[0]:.2e} "
                f"time={time.time() - t0:.0f}s")
        rec = {"epoch": ep + 1, "ce": tot_ce / n, "tri": tot_tri / n, "acc": tot_acc / n}

        if val_q is not None and (ep + 1) % 5 == 0 or ep == cfg.epochs - 1:
            if val_q is not None:
                m = validate(model, val_q, val_g, cfg, device)
                line += (f" | mAP={m['mAP']:.4f} R1={m['Rank-1']:.4f} R5={m['Rank-5']:.4f} "
                         f"mINP={m['mINP']:.4f} refF1={m['refusal_bestF1']:.3f} "
                         f"thr={m['refusal_threshold']:.3f}")
                rec.update(m)
                cands = [("обычные", model.state_dict(), m)]
                if ema_model is not None:
                    me = validate(ema_model, val_q, val_g, cfg, device)
                    line += f" | EMA mAP={me['mAP']:.4f}"
                    cands.append(("EMA", ema_model.state_dict(), me))
                which, sd_best, m_best = max(cands, key=lambda c: c[2]["mAP"])
                if m_best["mAP"] > best_map:
                    best_map = m_best["mAP"]
                    torch.save({"model": sd_best, "cfg": vars(cfg),
                                "num_classes": ds.num_classes, "metrics": m_best,
                                "weights_kind": which},
                               run_dir / "best.pt")
        print(line, flush=True)
        history.append(rec)
        # Чекпойнт для возобновления содержит и состояние оптимизатора, поэтому
        # весит втрое больше модели (для ViT-H+ это 5 ГБ). Каждую эпоху его
        # писать нельзя: за прогон набегает 150 ГБ записи, обучение заметно
        # тормозится о диск. Пишем раз в 5 эпох и в конце.
        if (ep + 1) % 5 == 0 or ep == cfg.epochs - 1:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep, "cfg": vars(cfg), "num_classes": ds.num_classes},
                       run_dir / "last.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=1, default=str))

    # финальный чекпоинт для инференса (только веса модели)
    torch.save({"model": model.state_dict(), "cfg": {k: str(v) for k, v in vars(cfg).items()},
                "num_classes": ds.num_classes}, run_dir / "final.pt")
    print("training done; saved", run_dir / "final.pt")


if __name__ == "__main__":
    main()
