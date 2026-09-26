"""Датасеты и сэмплер для обучения Re-ID."""
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T


def _hw(img_size):
    """Размер входа: число — квадрат, пара (h, w) — прямоугольник."""
    return (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)


class Letterbox:
    """Вписать кроп в (h, w) С СОХРАНЕНИЕМ пропорций и дополнить полями.

    Обычный Resize растягивает кроп в квадрат. Для типичного кропа (w/h 1.34)
    это терпимо, но у камер с вытянутыми кропами (w/h до 2.1) машина
    сжимается по ширине вдвое — разбор ошибок показал, что на такой камере
    проваливается 89% запросов. Поля заливаются средним цветом ImageNet,
    чтобы после нормировки они стали нулями.
    """

    FILL = (124, 116, 104)

    def __init__(self, size):
        self.h, self.w = size

    def __call__(self, img):
        s = min(self.w / img.width, self.h / img.height)
        nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
        img = img.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (self.w, self.h), self.FILL)
        canvas.paste(img, ((self.w - nw) // 2, (self.h - nh) // 2))
        return canvas


def build_transforms(img_size, mean, std, train: bool, aug: str = "base",
                     letterbox: bool = False):
    """aug: 'base' — исходный набор; 'shift[N]' — + Pad/RandomCrop со сдвигом N%
    (стандарт ReID); 'rrc' — RandomResizedCrop вместо Resize. Суффикс '+night'
    добавляет затемнение под ночные камеры.
    letterbox — вписывать с сохранением пропорций вместо растягивания."""
    h, w = _hw(img_size)
    night = aug.endswith("+night")
    aug = aug.replace("+night", "")
    resize = (Letterbox((h, w)) if letterbox
              else T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC))
    if not train:
        return T.Compose([resize, T.ToTensor(), T.Normalize(mean, std)])
    if aug == "rrc":
        geom = [T.RandomResizedCrop((h, w), scale=(0.7, 1.0), ratio=(0.85, 1.18),
                                    interpolation=T.InterpolationMode.BICUBIC)]
    elif aug.startswith("shift"):
        # классика ReID (Luo et al.): расширить и случайно вырезать обратно —
        # учит устойчивости к неточной рамке детектора. Суффикс задаёт долю
        # сдвига в процентах: shift = 5%, shift10 = 10%.
        frac = int(aug[5:] or 5) / 100
        pad = max(8, round(min(h, w) * frac))
        geom = [resize, T.Pad(pad, padding_mode="edge"), T.RandomCrop((h, w))]
    else:
        geom = [resize]
    photo = [T.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.15, hue=0.02)]
    if night:
        # ночная камера в данных вдвое темнее средней (яркость 58 против 109)
        # и проваливается втрое чаще; учим модель на искусственно тёмных кадрах
        photo.append(T.RandomApply([T.ColorJitter(brightness=(0.3, 0.7),
                                                  contrast=(0.6, 1.0))], p=0.3))
    return T.Compose(geom + [T.RandomHorizontalFlip(0.5)] + photo + [
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.2))], p=0.15),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.5, scale=(0.02, 0.2), value="random"),
    ])


class CropDataset(Dataset):
    """Читает заранее вырезанные кропы; отдаёт (img, label, camera)."""

    def __init__(self, df: pd.DataFrame, crops_dir: Path, transform, with_labels=True,
                 return_index=False):
        self.df = df.reset_index(drop=True)
        self.crops_dir = Path(crops_dir)
        self.transform = transform
        self.with_labels = with_labels
        self.return_index = return_index      # нужен дистилляции: индекс -> вектор учителя
        if with_labels:
            # плотные метки 0..C-1 для классификационной головы
            uniq = sorted(df["vehicle_id"].unique())
            self.vid2label = {v: i for i, v in enumerate(uniq)}
            self.num_classes = len(uniq)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # внешние датасеты дают готовый абсолютный путь к кропу
        path = getattr(r, "image_path", None)
        if isinstance(path, str) and path:
            img = Image.open(path).convert("RGB")
        else:
            img = Image.open(self.crops_dir / f"{r.image_id}.jpg").convert("RGB")
        img = self.transform(img)
        if self.with_labels:
            try:
                cam = int(r.camera_id)
            except (TypeError, ValueError):
                cam = -1          # внешние камеры (строковые) в обучении не используются
            if self.return_index:
                return img, self.vid2label[r.vehicle_id], cam, i
            return img, self.vid2label[r.vehicle_id], cam
        return img, 0, -1


class PKSampler(Sampler):
    """Идентично-сбалансированный сэмплер: P идентичностей x K снимков.

    Гарантирует, что в каждом батче у triplet-лосса есть валидные
    позитивные и негативные пары.
    """

    def __init__(self, labels, p: int, k: int, seed: int = 42, cameras=None):
        self.labels = np.asarray(labels)
        self.p, self.k = p, k
        self.rng = random.Random(seed)
        self.by_id = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            self.by_id[int(lab)].append(idx)
        self.ids = list(self.by_id)
        self.batches_per_epoch = len(self.labels) // (p * k)
        # камеро-осознанный режим: K снимков берутся с РАЗНЫХ камер, пока они
        # есть. Разбор ошибок показал, что провалы — это смена ракурса, а не
        # похожие машины, поэтому тяжёлые позитивы должны быть кросс-камерными.
        self.by_id_cam = None
        if cameras is not None:
            cams = np.asarray(cameras)
            self.by_id_cam = defaultdict(lambda: defaultdict(list))
            for idx, (lab, cam) in enumerate(zip(self.labels, cams)):
                self.by_id_cam[int(lab)][int(cam)].append(idx)

    def _take(self, pid):
        if self.by_id_cam is None:
            pool = self.by_id[pid]
            return (self.rng.sample(pool, self.k) if len(pool) >= self.k
                    else self.rng.choices(pool, k=self.k))
        cams = list(self.by_id_cam[pid])
        self.rng.shuffle(cams)
        out, ci = [], 0
        used = defaultdict(set)
        while len(out) < self.k:
            cam = cams[ci % len(cams)]
            ci += 1
            pool = [i for i in self.by_id_cam[pid][cam] if i not in used[cam]]
            if not pool:                       # камера исчерпана — можно повторять
                pool = self.by_id_cam[pid][cam]
            pick = self.rng.choice(pool)
            used[cam].add(pick)
            out.append(pick)
        return out

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            batch = []
            for pid in self.rng.sample(self.ids, self.p):
                batch.extend(self._take(pid))
            yield from batch

    def __len__(self):
        return self.batches_per_epoch * self.p * self.k


def split_train_val(df: pd.DataFrame, val_fraction: float, seed: int):
    """Разбиение по vehicle_id (open-set: валидационные ТС не видны при обучении)."""
    rng = random.Random(seed)
    ids = sorted(df["vehicle_id"].unique())
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    return df[~df.vehicle_id.isin(val_ids)].copy(), df[df.vehicle_id.isin(val_ids)].copy()


def build_val_protocol(val_df: pd.DataFrame, distractor_fraction: float, seed: int):
    """Собирает из val-части query/gallery так же, как устроен тест организаторов.

    Для каждой идентичности камеры делятся: часть снимков уходит в галерею,
    часть — в запросы. Доля distractor_fraction идентичностей в галерею не
    попадает вовсе — их запросы обязаны получить «отказ» (open-set негативы).
    """
    rng = random.Random(seed)
    ids = sorted(val_df["vehicle_id"].unique())
    rng.shuffle(ids)
    n_distr = int(len(ids) * distractor_fraction)
    distractor_ids = set(ids[:n_distr])
    return _split_by_camera(val_df, distractor_ids, rng)


def build_val_protocol_imagewise(val_df: pd.DataFrame, distractor_fraction: float,
                                 seed: int, gallery_fraction: float = 0.4):
    """Протокол, устроенный как тестовая выборка организаторов.

    Основной `build_val_protocol` делит камеры каждого ТС между галереей и
    запросами, поэтому снимок того же ТС с той же камеры в галерею попасть не
    может. В тестовой галерее такие кадры есть — её распределение близости
    бимодально с модой 0.93 (кадры-дубли). Здесь снимки делятся ПОКАДРОВО,
    без учёта камер, что воспроизводит реальную картину и нужно для честной
    калибровки порога отказа.
    """
    rng = random.Random(seed)
    ids = sorted(val_df["vehicle_id"].unique())
    rng.shuffle(ids)
    distractor_ids = set(ids[: int(len(ids) * distractor_fraction)])
    q_rows, g_rows = [], []
    for vid, grp in val_df.groupby("vehicle_id"):
        if vid in distractor_ids or len(grp) < 2:
            q_rows.append(grp)
            continue
        idx = list(range(len(grp)))
        rng.shuffle(idx)
        n_g = max(1, int(round(len(grp) * gallery_fraction)))
        g_rows.append(grp.iloc[idx[:n_g]])
        q_rows.append(grp.iloc[idx[n_g:]])
    return (pd.concat(q_rows).reset_index(drop=True),
            pd.concat(g_rows).reset_index(drop=True))


def _split_by_camera(val_df, distractor_ids, rng):
    q_rows, g_rows = [], []
    for vid, grp in val_df.groupby("vehicle_id"):
        cams = sorted(grp["camera_id"].unique())
        if vid in distractor_ids or len(cams) < 2:
            q_rows.append(grp)          # весь ТС только в запросах -> в галерее пары нет
            continue
        rng.shuffle(cams)
        g_cams = set(cams[: max(1, len(cams) // 2)])
        g_rows.append(grp[grp.camera_id.isin(g_cams)])
        q_rows.append(grp[~grp.camera_id.isin(g_cams)])
    return pd.concat(q_rows).reset_index(drop=True), pd.concat(g_rows).reset_index(drop=True)
