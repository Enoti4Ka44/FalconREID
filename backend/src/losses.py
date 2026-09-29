"""Функции потерь: batch-hard triplet (Hermans et al., 2017)."""
import torch
import torch.nn.functional as F


def batch_hard_triplet(feat: torch.Tensor, labels: torch.Tensor, margin: float = 0.3):
    """Для каждого якоря берём самый дальний позитив и самый близкий негатив."""
    dist = torch.cdist(feat, feat, p=2)                      # (B, B)
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=feat.device)
    pos = dist.masked_fill(~same | eye, float("-inf")).max(1).values
    neg = dist.masked_fill(same, float("inf")).min(1).values
    valid = torch.isfinite(pos) & torch.isfinite(neg)
    if not valid.any():
        return feat.new_zeros(())
    return F.relu(pos[valid] - neg[valid] + margin).mean()


class CrossBatchMemory:
    """Память признаков из предыдущих батчей (XBM, Wang et al., CVPR 2020).

    У крупных моделей батч маленький (ViT-H+ на 16 ГБ — 6 машин × 4 снимка),
    и batch-hard triplet видит всего 20 негативов на якорь. Разбор ошибок
    показал, что половина провалов — «почти угадал»: верный кандидат на
    волосок ниже ложного. Такие случаи лечатся именно трудными негативами,
    а их в маленьком батче мало. Память хранит признаки последних N снимков
    (без градиента) и добавляет их как кандидатов в позитивы и негативы.

    Работает, потому что backbone учится медленно (LR ×0.1, нижние блоки
    заморожены): признаки в памяти устаревают незначительно.
    """

    def __init__(self, size: int, dim: int, device):
        self.size = size
        self.feats = torch.zeros(size, dim, device=device)
        self.labels = torch.full((size,), -1, dtype=torch.long, device=device)
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def enqueue(self, feat, labels):
        n = feat.shape[0]
        idx = (torch.arange(n, device=feat.device) + self.ptr) % self.size
        self.feats[idx] = feat.detach().float()
        self.labels[idx] = labels
        self.ptr = (self.ptr + n) % self.size
        self.full = self.full or self.ptr < n

    def get(self):
        if self.full:
            return self.feats, self.labels
        return self.feats[: self.ptr], self.labels[: self.ptr]


def xbm_triplet(feat, labels, mem_feat, mem_labels, margin: float = 0.3):
    """Batch-hard triplet: якоря из батча, кандидаты — из памяти."""
    if mem_feat.shape[0] == 0:
        return feat.new_zeros(())
    dist = torch.cdist(feat, mem_feat, p=2)
    same = labels[:, None].eq(mem_labels[None, :])
    pos = dist.masked_fill(~same, float("-inf")).max(1).values
    neg = dist.masked_fill(same, float("inf")).min(1).values
    valid = torch.isfinite(pos) & torch.isfinite(neg)
    if not valid.any():
        return feat.new_zeros(())
    return F.relu(pos[valid] - neg[valid] + margin).mean()


def similarity_kd(student, teacher, temperature: float = 0.1):
    """Дистилляция структуры сходств внутри батча (relational KD).

    Для каждого снимка батча сравниваются распределения «на кого из батча он
    похож» у учителя и студента: softmax по косинусам с температурой, KL
    между ними. Штрафуется именно неверный ПОРЯДОК соседей — то, что меряет
    mAP@10. Размерности учителя и студента могут не совпадать, проекционная
    голова не нужна.
    """
    s = F.normalize(student.float(), dim=1)
    t = F.normalize(teacher.float(), dim=1)
    n = s.shape[0]
    mask = torch.eye(n, dtype=torch.bool, device=s.device)
    logit_s = (s @ s.T).masked_fill(mask, float("-inf")) / temperature
    logit_t = (t @ t.T).masked_fill(mask, float("-inf")) / temperature
    log_p_s = F.log_softmax(logit_s, dim=1)
    p_t = F.softmax(logit_t, dim=1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-12)) - log_p_s)).masked_fill(mask, 0).sum(1)
    return kl.mean() * temperature ** 2
