"""Инференс на тестовой выборке: embeddings.npy, submission.csv, candidates.csv.

Запуск: python -m src.infer --ckpt runs/run/final.pt --threshold 0.377
Порядок embeddings.npy: сначала все image_id из test_query.csv (по порядку
файла), затем все image_id из test_gallery.csv — как требует README датасета.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .train import extract_embeddings
from .model import ReIDModel, ReIDModelParts


def load_model(ckpt_path: str, cfg: Config, device: str):
    # веса могут храниться в int8 (src/int8_pack.py) — распаковка в fp16
    # происходит один раз при загрузке, арифметика инференса не меняется
    from .int8_pack import load_packed
    ck = load_packed(Path(ckpt_path))
    sd = ck["model"]
    # тип модели определяется по чекпойнту: части-признаки несут part_proj.*
    model_cls = ReIDModelParts if any(k.startswith("part_proj.") for k in sd) else ReIDModel
    model = model_cls(cfg.backbone, ck["num_classes"], cfg.embed_dim, pretrained=False,
                      img_size=cfg.img_size)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, required=True,
                    help="порог косинусной близости для режима отказа (подобран на валидации)")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--topk", type=int, default=10)
    # вторая модель ансамбля (конкатенация эмбеддингов с весом w2)
    ap.add_argument("--ckpt2", default=None)
    ap.add_argument("--backbone2", default="convnext_base.fb_in22k_ft_in1k")
    ap.add_argument("--img-size2", type=int, default=224)
    ap.add_argument("--w2", type=float, default=0.5)
    a = ap.parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q_df = pd.read_csv(cfg.data_root / "test_query.csv", dtype={"image_id": str})
    g_df = pd.read_csv(cfg.data_root / "test_gallery.csv", dtype={"image_id": str})

    from .ensemble import MANIFEST, extract_ensemble, load_manifest, weights_size_mb
    if MANIFEST.exists():
        # ансамбль из манифеста (единая точка правды для всех компонентов)
        manifest = load_manifest()
        print(f"ансамбль из {MANIFEST}: {len(manifest)} моделей, "
              f"веса файлов {weights_size_mb(manifest):.0f} МБ (лимит 2048)")
        # единый экстрактор: полный кадр с диска -> вектор, тот же код, что
        # меряется на скорость (src/extractor.py); кэш кропов не нужен
        from .extractor import Extractor
        ex = Extractor(device=device)
        q_emb = ex.extract_frame(q_df, cfg.data_root / "images")
        g_emb = ex.extract_frame(g_df, cfg.data_root / "images")
        del ex
        print(f"dim={q_emb.shape[1]}")
    else:
        model = load_model(a.ckpt, cfg, device)
        q_emb = extract_embeddings(model, q_df, cfg, device)
        g_emb = extract_embeddings(model, g_df, cfg, device)
        if a.ckpt2:
            del model
            torch.cuda.empty_cache() if device == "cuda" else None
            cfg2 = Config()
            cfg2.backbone, cfg2.img_size = a.backbone2, a.img_size2
            model2 = load_model(a.ckpt2, cfg2, device)
            q2 = extract_embeddings(model2, q_df, cfg2, device)
            g2 = extract_embeddings(model2, g_df, cfg2, device)
            q_emb = np.concatenate([q_emb, a.w2 * q2], axis=1)
            g_emb = np.concatenate([g_emb, a.w2 * g2], axis=1)
            q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
            g_emb /= np.linalg.norm(g_emb, axis=1, keepdims=True)
            print(f"ensemble: + {a.backbone2} (w2={a.w2}), dim={q_emb.shape[1]}")

    # постобработка эмбеддингов (DBA + alphaQE) — параметры подобраны на
    # валидации (models/postproc.json); итоговые векторы сохраняются в
    # embeddings.npy, submission строго согласован с ними
    pp_file = Path(a.ckpt).parent / "postproc.json"
    if not pp_file.exists():
        pp_file = Path("models/postproc.json")
    if pp_file.exists():
        from .postproc_experiments import aqe, dba
        pp = json.loads(pp_file.read_text(encoding="utf-8"))
        if pp.get("dba_k"):
            g_emb = dba(g_emb, k=pp["dba_k"], alpha=pp.get("alpha", 3.0))
        if pp.get("aqe_k"):
            q_emb = aqe(q_emb, g_emb, k=pp["aqe_k"], alpha=pp.get("alpha", 3.0))
        print(f"postproc: DBA k={pp.get('dba_k')} + AQE k={pp.get('aqe_k')}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    # 1) embeddings.npy: query потом gallery, порядок строго по CSV
    np.save(a.out_dir / "embeddings.npy",
            np.concatenate([q_emb, g_emb]).astype(np.float32))

    sim = q_emb @ g_emb.T
    order = np.argsort(-sim, axis=1)

    # 2) submission.csv: топ-10 кандидатов на запрос (чистое ранжирование,
    #    строго согласованное с embeddings.npy; одно-камерные совпадения
    #    исключает эталонный скрипт организаторов)
    gallery_ids = g_df["image_id"].values
    with open(a.out_dir / "submission.csv", "w", newline="") as f:
        f.write("query_id," + ",".join(f"gallery_id_{i+1}" for i in range(a.topk)) + "\n")
        for qi, qid in enumerate(q_df["image_id"]):
            top = gallery_ids[order[qi, :a.topk]]
            f.write(qid + "," + ",".join(top) + "\n")

    # 3) candidates.csv: продуктовый режим. Суть ReID — КРОСС-камерные
    #    сопоставления, поэтому кадры-дубли той же сцены, что и запрос
    #    (почти тот же фон кадра; правило отсеивает дубли с точностью
    #    0.97–1.0, но не восстанавливает камеры целиком — см. docs/PROGRESS.md,
    #    дефект Д3), исключаются; порог применяется к оставшимся.
    #    Нет уверенного кросс-камерного кандидата — отказ.
    cam_file = a.out_dir / "test_cameras.csv"
    # кэш годится только для ТОЙ ЖЕ тестовой выборки: на закрытом тесте жюри
    # image_id другие (устаревший файл дал бы KeyError), а группы — связные
    # компоненты по всем кадрам, поэтому даже надмножество даёт другие группы
    need_ids = set(q_df.image_id) | set(g_df.image_id)
    cached = (set(pd.read_csv(cam_file, dtype={"image_id": str}).image_id)
              if cam_file.exists() else set())
    if cam_file.exists() and need_ids != cached:
        print(f"{cam_file} построен для другой выборки — пересчитываю")
        cam_file.unlink()
    if not cam_file.exists():
        from .camera_infer import main as cam_main
        import sys as _sys
        argv_bak = _sys.argv
        _sys.argv = ["camera_infer", "--thr", "0.80", "--out", str(cam_file)]
        cam_main()
        _sys.argv = argv_bak
    cams = pd.read_csv(cam_file, dtype={"image_id": str})
    grp = dict(zip(cams.image_id, cams.camera_group))
    q_grp = np.array([grp[i] for i in q_df.image_id])
    g_grp = np.array([grp[i] for i in g_df.image_id])
    same_scene = q_grp[:, None] == g_grp[None, :]
    # расширение: «та же парковка» — умеренно похожий фон + почти тот же BBox
    # (машина стоит на месте, кадры одной камеры в разное время; правило
    # проверено на валидации — не ухудшает F1/TNR, на train precision 0.97)
    from .camera_infer import compute_descriptors
    from .camera_rule_val import bbox_iou
    dq = compute_descriptors(q_df, cfg)
    dg = compute_descriptors(g_df, cfg)
    bsim = dq @ dg.T
    iou = bbox_iou(q_df[["x", "y", "w", "h"]].values.astype(float),
                   g_df[["x", "y", "w", "h"]].values.astype(float))
    same_scene |= (bsim >= 0.50) & (iou >= 0.85)
    same_scene |= (bsim >= 0.30) & (iou >= 0.90)

    n_refused = 0
    with open(a.out_dir / "candidates.csv", "w", newline="") as f:
        f.write("query_id,gallery_id,confidence\n")
        for qi, qid in enumerate(q_df["image_id"]):
            accepted = [(gallery_ids[j], sim[qi, j]) for j in order[qi]
                        if not same_scene[qi, j] and sim[qi, j] >= a.threshold][: a.topk]
            if not accepted:
                # отказ = ОТСУТСТВИЕ строк для query_id (ответ постановщика,
                # вопросы 18/20/29); строка с пустым gallery_id не годится
                n_refused += 1
            else:
                for gid, s in accepted:
                    f.write(f"{qid},{gid},{s:.4f}\n")

    sim_x = np.where(same_scene, -1.0, sim)
    stats = {"queries": len(q_df), "gallery": len(g_df), "refused": n_refused,
             "accept_rate": 1 - n_refused / len(q_df), "threshold": a.threshold,
             "max_sim_mean_raw": float(sim.max(1).mean()),
             "max_sim_mean_crosscam": float(sim_x.max(1).mean()),
             "queries_with_samescene_top1": float((sim.max(1) - sim_x.max(1) > 1e-6).mean())}
    (a.out_dir / "inference_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
