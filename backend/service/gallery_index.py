"""GalleryIndex — работа с галереей: FAISS, эмбеддинги, статистика.

Не использует torch. Загружает gallery_index.npz и постобработку.
"""
import collections
from pathlib import Path

import numpy as np
from PIL import Image


class GalleryIndex:
    """Индекс галереи: FAISS-поиск, статистика, пути к изображениям."""

    def __init__(self, gallery_index: Path, data_root: Path, threshold: float):
        self.data_root = Path(data_root)
        self.crops_dir = self.data_root / "crops"

        idx = np.load(gallery_index, allow_pickle=True)
        self.gallery_ids = idx["ids"]
        emb = np.ascontiguousarray(idx["emb"].astype(np.float32))
        self.gallery_mat = emb
        self.gallery_groups = (
            idx["groups"] if "groups" in idx.files
            else np.zeros(len(self.gallery_ids), np.int64)
        )
        self.id2idx = {str(g): i for i, g in enumerate(self.gallery_ids)}

        import faiss
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.gallery_size = len(self.gallery_ids)

        self.threshold = threshold

        pp_file = Path(gallery_index).parent / "postproc.json"
        self.aqe_k, self.aqe_alpha = 0, 3.0
        if pp_file.exists():
            import json
            pp = json.loads(pp_file.read_text(encoding="utf-8"))
            self.aqe_k = int(pp.get("aqe_k") or 0)
            self.aqe_alpha = float(pp.get("alpha", 3.0))

        self._tracks_cache = None

    def search(self, emb: np.ndarray, top_k: int = 10, threshold=None):
        thr = self.threshold if threshold is None else threshold
        if self.aqe_k:
            sim = self.gallery_mat @ emb
            nn = np.argsort(-sim)[: self.aqe_k]
            w = np.clip(sim[nn], 0, None) ** self.aqe_alpha
            emb = emb + (w[:, None] * self.gallery_mat[nn]).sum(0)
            emb = emb / np.linalg.norm(emb)
        sims_k, ids_k = self.index.search(emb[None, :].astype(np.float32), top_k)
        cands = [
            {
                "gallery_id": str(self.gallery_ids[j]),
                "confidence": round(float(s), 4),
                "accepted": bool(s >= thr),
                "camera_group": int(self.gallery_groups[j]),
            }
            for s, j in zip(sims_k[0], ids_k[0]) if j >= 0
        ]
        all_sims = self.gallery_mat @ emb.astype(np.float32)
        n_acc = sum(c["accepted"] for c in cands)
        hist, edges = np.histogram(all_sims, bins=48, range=(-0.2, 1.0))
        return {
            "refused": n_acc == 0,
            "n_accepted": n_acc,
            "candidates": cands,
            "sim_hist": {
                "counts": hist.tolist(),
                "edges": [round(float(e), 3) for e in edges],
            },
            "sim_max": round(float(all_sims.max()), 4),
        }

    def similar_in_gallery(self, image_id: str, top_k: int = 12):
        i = self.id2idx.get(image_id)
        if i is None:
            return None
        sims = self.gallery_mat @ self.gallery_mat[i]
        order = np.argsort(-sims)
        out = []
        for j in order:
            if j == i:
                continue
            out.append({
                "gallery_id": str(self.gallery_ids[j]),
                "confidence": round(float(sims[j]), 4),
                "camera_group": int(self.gallery_groups[j]),
                "same_scene": bool(
                    self.gallery_groups[j] == self.gallery_groups[i]
                    and self.gallery_groups[i] >= 0
                ),
            })
            if len(out) >= top_k:
                break
        return {
            "gallery_id": image_id,
            "camera_group": int(self.gallery_groups[i]),
            "neighbors": out,
        }

    def location_stats(self):
        cnt = collections.Counter(int(g) for g in self.gallery_groups)
        return sorted(
            ({"camera_group": g, "count": c} for g, c in cnt.items()),
            key=lambda x: -x["count"],
        )

    def gallery_page(self, group: int | None = None, offset: int = 0, limit: int = 60):
        idxs = (
            range(self.gallery_size) if group is None
            else [i for i in range(self.gallery_size) if self.gallery_groups[i] == group]
        )
        idxs = list(idxs)
        page = idxs[offset:offset + limit]
        return {
            "total": len(idxs),
            "offset": offset,
            "items": [
                {
                    "gallery_id": str(self.gallery_ids[i]),
                    "camera_group": int(self.gallery_groups[i]),
                }
                for i in page
            ],
        }

    def cross_location_tracks(self, thr: float = 0.65, min_locs: int = 2):
        if self._tracks_cache is not None:
            return self._tracks_cache
        n = self.gallery_size
        sim = self.gallery_mat @ self.gallery_mat.T
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(n):
            for j in np.where(sim[i, i + 1:] >= thr)[0] + i + 1:
                ra, rb = find(i), find(int(j))
                if ra != rb:
                    parent[rb] = ra
        comps = {}
        for i in range(n):
            comps.setdefault(find(i), []).append(i)
        tracks = []
        for members in comps.values():
            locs = sorted({int(self.gallery_groups[i]) for i in members})
            if len(members) >= 2 and len(locs) >= min_locs:
                tracks.append({
                    "size": len(members),
                    "locations": locs,
                    "items": [
                        {
                            "gallery_id": str(self.gallery_ids[i]),
                            "camera_group": int(self.gallery_groups[i]),
                        }
                        for i in members[:8]
                    ],
                })
        tracks.sort(key=lambda t: (-len(t["locations"]), -t["size"]))
        self._tracks_cache = {
            "threshold": thr,
            "n_tracks": len(tracks),
            "tracks": tracks[:60],
        }
        return self._tracks_cache

    def pair_matrix(self, ids: list[str]):
        idxs = [self.id2idx.get(i) for i in ids]
        if any(i is None for i in idxs):
            return None
        sub = self.gallery_mat[idxs]
        return np.round(sub @ sub.T, 4).tolist()

    def vector_of(self, gallery_id: str):
        i = self.id2idx.get(gallery_id)
        return None if i is None else self.gallery_mat[i]

    def thumb_path(self, image_id: str):
        p = self.crops_dir / f"{image_id}.jpg"
        return p if p.exists() else None

    def gallery_crop(self, image_id: str):
        p = self.thumb_path(image_id)
        return Image.open(p).convert("RGB") if p else None
