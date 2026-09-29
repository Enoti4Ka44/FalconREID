"""Сервис детекции ТС (YOLOv8n)."""
from PIL import Image

from ..exceptions import NotFoundError


class DetectorService:
    def __init__(self) -> None:
        self._model = None

    def _load(self, model_path: str) -> None:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(model_path)

    def detect(self, img: Image.Image, model_path: str) -> dict:
        self._load(model_path)
        # Два масштаба: YOLOv8n на одном размере входа иногда пропускает крупную
        # тёмную машину ночью (замер на тестовых кадрах: 640 — 28/30, 960 — 30/30
        # целевых ТС, но отдельные ночные кадры видит только 640). Рамки
        # объединяются, дубли (IoU > 0.6) отбрасываются.
        boxes = []
        for size in (640, 960):
            res = self._model.predict(
                img, conf=0.28, imgsz=size, classes=[2, 3, 5, 7],
                device="cpu", verbose=False,
            )[0]
            for b in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                cand = {
                    "x": round(x1), "y": round(y1),
                    "w": round(x2 - x1), "h": round(y2 - y1),
                    "score": round(float(b.conf[0]), 3),
                    "cls": res.names[int(b.cls[0])],
                }
                if all(_iou(cand, o) <= 0.6 for o in boxes):
                    boxes.append(cand)
        boxes.sort(key=lambda b: -b["w"] * b["h"])
        return {"width": img.width, "height": img.height, "boxes": boxes[:20]}


def _iou(a: dict, b: dict) -> float:
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    return inter / max(a["w"] * a["h"] + b["w"] * b["h"] - inter, 1)
