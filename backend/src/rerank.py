"""k-reciprocal re-ranking в потоковом режиме (по мотивам Zhong et al., 2017).

Постановщик разрешил переранжирование в `submission.csv`, но только «не
использующее другие query»: протокол потоковый, решению доступны вся
галерея (статичная база, её можно готовить заранее) и ОДИН текущий запрос.
Классический алгоритм строит матрицу расстояний по всем запросам сразу —
так нельзя.

Поэтому алгоритм разделён на две части:

* `GalleryIndex` — всё, что зависит только от галереи: соседи, k-взаимные
  множества, веса. Строится один раз, как индекс FAISS.
* `GalleryIndex.distances(q)` — один запрос против готового индекса.
  Ничего о других запросах эта функция не знает, поэтому потоковость
  гарантирована самой конструкцией, а не дисциплиной вызова.

Итоговое расстояние: d* = (1 − λ)·d_jaccard + λ·d_original.
"""
import numpy as np


class GalleryIndex:
    def __init__(self, G, k1=10, k2=3):
        G = np.asarray(G, dtype=np.float32)
        self.G = G
        self.k1, self.k2 = k1, k2
        n = len(G)
        D = np.clip(2.0 - 2.0 * (G @ G.T), 0.0, None)
        self.rank = np.argsort(D, axis=1)                 # [:, 0] — сам объект
        k1 = min(k1, n - 1)
        self.k1 = k1
        # расстояние до k1-го соседа: запрос входит в окрестность объекта
        # галереи, если он ближе этого соседа
        self.kth = D[np.arange(n), self.rank[:, k1]]
        half = max(1, int(round(k1 / 2)))
        self.half_rec = [self._recip(i, half) for i in range(n)]

        V = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            rec = self._recip(i, k1)
            exp = self._expand(rec)
            w = np.exp(-D[i, exp])
            V[i, exp] = w / max(w.sum(), 1e-12)
        if k2 > 1:
            V = np.stack([V[self.rank[i, :k2]].mean(0) for i in range(n)])
        self.V = V

    def _recip(self, i, k):
        fwd = self.rank[i, : k + 1]
        back = self.rank[fwd, : k + 1]
        return fwd[np.any(back == i, axis=1)]

    def _expand(self, rec):
        out = [rec]
        rs = set(rec.tolist())
        for c in rec:
            cr = self.half_rec[c]
            if len(rs.intersection(cr.tolist())) > 2 / 3 * len(cr):
                out.append(cr)
        return np.unique(np.concatenate(out))

    def distances(self, q, lam=0.3):
        """Переранжированные расстояния ОДНОГО запроса до всей галереи."""
        q = np.asarray(q, dtype=np.float32)
        dq = np.clip(2.0 - 2.0 * (self.G @ q), 0.0, None)
        order = np.argsort(dq)
        nq = order[: self.k1]
        rec = nq[dq[nq] <= self.kth[nq]]                  # взаимные соседи запроса
        if len(rec) == 0:
            rec = nq[:1]
        exp = self._expand(rec)
        vq = np.zeros(len(self.G), dtype=np.float32)
        w = np.exp(-dq[exp])
        vq[exp] = w / max(w.sum(), 1e-12)
        if self.k2 > 1:
            vq = (vq + self.V[order[: self.k2 - 1]].sum(0)) / self.k2
        mins = np.minimum(vq[None, :], self.V).sum(1)
        jac = 1.0 - mins / (2.0 - mins)
        return (1 - lam) * jac + lam * dq / max(dq.max(), 1e-12)

    def rank_all(self, Q, lam=0.3, topk=None):
        """Порядок галереи для каждого запроса — каждый считается отдельно."""
        out = []
        for q in Q:
            o = np.argsort(self.distances(q, lam))
            out.append(o if topk is None else o[:topk])
        return np.stack(out)
