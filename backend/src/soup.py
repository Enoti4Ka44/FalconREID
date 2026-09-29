"""Усреднение весов нескольких обучений одной модели (model soup).

Несколько запусков с разными сидами сходятся в один бассейн, и среднее их
весов обычно работает лучше любого отдельного. Инференс при этом не
замедляется и лимит весов не растёт — в отличие от ансамбля.

Жадный вариант (`--greedy`) добавляет модель в суп только если это НЕ
ухудшает mAP на замороженном эталоне (Wortsman et al.), поэтому неудачный
прогон не портит результат.

    python -m src.soup --out runs/soup_b/best.pt runs/holdout_b_s*/best.pt
    python -m src.soup --greedy --backbone <timm> --img-size 252 \
        --out runs/soup_b/best.pt runs/holdout_b_s*/best.pt
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dataset import build_val_protocol, split_train_val
from .eval_utils import compute_reid_metrics
from .postproc_experiments import aqe, dba, norm


def average(state_dicts):
    out = copy.deepcopy(state_dicts[0])
    for k, v in out.items():
        if not v.is_floating_point():
            continue
        acc = torch.zeros_like(v, dtype=torch.float64)
        for sd in state_dicts:
            acc += sd[k].double()
        out[k] = (acc / len(state_dicts)).to(v.dtype)
    return out


def _score(sd, ck0, backbone, size, vq, vg, args):
    from .model import ReIDModel, ReIDModelParts
    from .train import extract_embeddings
    c = Config()
    c.backbone, c.img_size = backbone, size
    cls = ReIDModelParts if any(k.startswith("part_proj.") for k in sd) else ReIDModel
    m = cls(backbone, ck0["num_classes"], c.embed_dim, pretrained=False, img_size=size)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v
                       for k, v in sd.items()})
    m = m.cuda().eval()
    q = extract_embeddings(m, vq, c, "cuda", bs=32)
    g = extract_embeddings(m, vg, c, "cuda", bs=32)
    del m
    torch.cuda.empty_cache()
    g = dba(norm(g), k=4, alpha=1.0)
    q = aqe(norm(q), g, k=1, alpha=1.0)
    return compute_reid_metrics(q, g, *args)["mAP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--img-size", type=int, default=None)
    a = ap.parse_args()

    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in a.ckpts]
    sds = [c["model"] for c in cks]
    print(f"моделей в супе: {len(sds)}")

    if not a.greedy:
        merged = average(sds)
    else:
        cfg = Config()
        df = pd.read_csv(cfg.data_root / "train.csv")
        _, val_df = split_train_val(df, cfg.val_id_fraction, cfg.seed)
        vq, vg = build_val_protocol(val_df, cfg.distractor_fraction, cfg.seed)
        args = (vq.vehicle_id.values, vg.vehicle_id.values,
                vq.camera_id.values, vg.camera_id.values)
        solo = [(_score(sd, cks[0], a.backbone, a.img_size, vq, vg, args), i)
                for i, sd in enumerate(sds)]
        for s, i in sorted(solo, reverse=True):
            print(f"  соло {Path(a.ckpts[i]).parent.name}: mAP={s:.4f}")
        order = [i for _, i in sorted(solo, reverse=True)]
        chosen = [order[0]]
        best = solo[[j for _, j in solo].index(order[0])][0]
        for i in order[1:]:
            cand = average([sds[j] for j in chosen + [i]])
            s = _score(cand, cks[0], a.backbone, a.img_size, vq, vg, args)
            keep = s >= best
            print(f"  + {Path(a.ckpts[i]).parent.name}: mAP={s:.4f} "
                  f"{'берём' if keep else 'отбрасываем'}")
            if keep:
                chosen.append(i)
                best = s
        print(f"итоговый суп из {len(chosen)}: mAP={best:.4f}")
        merged = average([sds[j] for j in chosen])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ck = dict(cks[0])
    ck["model"] = merged
    ck["soup_of"] = a.ckpts
    torch.save(ck, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
