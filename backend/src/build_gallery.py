"""Построение индекса галереи для сервиса: эмбеддинги test_gallery -> gallery_index.npz."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .infer import load_model
from .train import extract_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="не нужен при наличии models/ensemble.json")
    ap.add_argument("--ckpt2", default=None)
    ap.add_argument("--backbone2", default="convnext_base.fb_in22k_ft_in1k")
    ap.add_argument("--img-size2", type=int, default=224)
    ap.add_argument("--w2", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("models/gallery_index.npz"))
    a = ap.parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g_df = pd.read_csv(cfg.data_root / "test_gallery.csv")

    from .ensemble import MANIFEST
    if MANIFEST.exists():
        # тот же экстрактор, что строит embeddings.npy и меряется на скорость
        from .extractor import Extractor
        emb = Extractor(device=device).extract_frame(g_df, cfg.data_root / "images")
        print(f"ensemble gallery (manifest) dim={emb.shape[1]}")
    else:
        model = load_model(a.ckpt, cfg, device)
        emb = extract_embeddings(model, g_df, cfg, device)
        if a.ckpt2:
            import torch as _t
            del model
            _t.cuda.empty_cache() if device == "cuda" else None
            cfg2 = Config()
            cfg2.backbone, cfg2.img_size = a.backbone2, a.img_size2
            model2 = load_model(a.ckpt2, cfg2, device)
            emb2 = extract_embeddings(model2, g_df, cfg2, device)
            emb = np.concatenate([emb, a.w2 * emb2], axis=1)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            print(f"ensemble gallery dim={emb.shape[1]}")
    # DBA-расширение галереи (та же постобработка, что в офлайн-инференсе)
    pp_file = Path("models/postproc.json")
    if pp_file.exists():
        import json
        from .postproc_experiments import dba
        pp = json.loads(pp_file.read_text(encoding="utf-8"))
        if pp.get("dba_k"):
            emb = dba(emb, k=pp["dba_k"], alpha=pp.get("alpha", 3.0))
            print(f"applied DBA k={pp['dba_k']}")
    # камеры-группы (восстановленные по фону кадра) — для карты перемещений в UI
    groups = np.zeros(len(g_df), np.int64)
    cam_file = Path("artifacts/test_cameras.csv")
    if cam_file.exists():
        cams = pd.read_csv(cam_file)
        grp = dict(zip(cams.image_id, cams.camera_group))
        groups = np.array([grp.get(i, -1) for i in g_df.image_id])
        print(f"camera groups attached: {len(set(groups))} групп")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, ids=g_df["image_id"].values.astype(str),
             emb=emb.astype(np.float32), groups=groups)
    print(f"gallery index: {len(g_df)} vectors dim={emb.shape[1]} -> {a.out}")


if __name__ == "__main__":
    main()
