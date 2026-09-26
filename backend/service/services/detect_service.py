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
        res = self._model.predict(
            img, conf=0.28, imgsz=1280, classes=[2, 3, 5, 7],
            device="cpu", verbose=False,
        )[0]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            boxes.append({
                "x": round(x1), "y": round(y1),
                "w": round(x2 - x1), "h": round(y2 - y1),
                "score": round(float(b.conf[0]), 3),
                "cls": res.names[int(b.cls[0])],
            })
        boxes.sort(key=lambda b: -b["w"] * b["h"])
        return {"width": img.width, "height": img.height, "boxes": boxes[:20]}
