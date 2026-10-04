"""Детекторы объектов и трекер игроков.

Все детекторы возвращают список `Detection` — общий формат, не зависящий от
модели. Основной баскетбольный детектор (Roboflow RF-DETR) работает в
отдельном процессе, см. perception.py; здесь — локальная YOLO, облачный API
Roboflow и трекер ByteTrack.
"""

from dataclasses import dataclass
import base64
import os
import re

import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple  # x1, y1, x2, y2 в пикселях кадра анализа
    confidence: float
    kind: str  # player | referee | ball | basket | hoop | number
    action: str = ""  # shot | layup | block | possession | made | ""
    track_id: int | None = None
    text: str = ""  # распознанный номер для kind == "number"
    label: str = ""  # исходное имя класса модели

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot(self):
        """Точка контакта с полом: середина нижней стороны рамки."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def classify(name):
    """Имя класса модели → (kind, action).

    Поддерживаются классы датасета basketball-player-detection-3 (Roboflow)
    и COCO (`person`, `sports ball`).
    """
    name = str(name).lower().replace("_", "-").replace(" ", "-")
    if name in ("sports-ball", "ball", "basketball"):
        return "ball", ""
    if name in ("ball-in-basket", "ball-in-hoop"):
        return "basket", "made"
    if name in ("hoop", "rim", "basket", "basketball-hoop"):
        return "hoop", ""
    if name in ("number", "jersey-number"):
        return "number", ""
    if name in ("referee", "ref"):
        return "referee", ""
    if name == "person" or name.startswith("player"):
        if "jump-shot" in name:
            return "player", "shot"
        if "layup" in name or "dunk" in name:
            return "player", "layup"
        if "shot-block" in name or name.endswith("-block"):
            return "player", "block"
        if "possession" in name:
            return "player", "possession"
        return "player", ""
    return "ignore", ""


def make_detection(label, box, confidence, text="", track_id=None):
    kind, action = classify(label)
    if kind == "ignore":
        return None
    return Detection(tuple(map(float, box)), float(confidence), kind, action,
                     track_id=None if track_id is None else int(track_id),
                     text=str(text or ""), label=str(label))


class ProvidedTracks:
    """Треки уже пришли из восприятия (SAM2 в отдельном окружении).

    Интерфейс совпадает с PlayerTracker, поэтому остальному конвейеру всё
    равно, кто вёл игроков: ByteTrack локально или SAM2 на этапе восприятия.
    """

    source = "sam2"

    def __init__(self, fps=None, id_offset=0):
        self.offset = id_offset

    def update(self, detections):
        return [
            Detection(d.box, d.confidence, "player", d.action, d.track_id + self.offset, d.text, d.label)
            for d in detections
            if d.kind == "player" and d.track_id is not None
        ]


def _torch_device(requested):
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU недоступен. Выберите GPU в Colab или режим CPU.")
    return requested


class YoloDetector:
    """Локальная YOLO (ultralytics): COCO-веса как базовая линия или свои веса."""

    provides_keypoints = False

    def __init__(self, cfg):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = _torch_device(cfg.device)
        self.model = YOLO(cfg.weights)
        names = {str(n).lower() for n in self.model.names.values()}
        # есть ли у модели баскетбольные классы действий (для логики бросков)
        self.has_actions = any("jump-shot" in n or "ball-in-basket" in n for n in names)

    def detect(self, frame, frame_index=0):
        try:
            result = self.model.predict(
                frame,
                imgsz=self.cfg.image_size,
                conf=self.cfg.confidence,
                iou=0.5,
                device=self.device,
                half=self.device.startswith("cuda"),
                verbose=False,
            )[0]
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError("Не хватает видеопамяти. Уменьшите размер изображения до 640.") from None
            raise
        output = []
        if result.boxes is not None:
            for box, score, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                detection = make_detection(result.names[int(class_id)], box, score)
                if detection is not None:
                    output.append(detection)
        return output


class YoloCourtKeypoints:
    """Свои веса YOLO-pose для ключевых точек площадки (33 точки в порядке court.py)."""

    def __init__(self, weights, device="auto", image_size=960):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = _torch_device(device)
        self.image_size = image_size

    def keypoints(self, frame, frame_index=0):
        result = self.model.predict(frame, imgsz=self.image_size, device=self.device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return None
        best = int(np.argmax(result.boxes.conf.cpu().numpy())) if result.boxes is not None else 0
        xy = result.keypoints.xy.cpu().numpy()[best]
        conf = result.keypoints.conf
        conf = conf.cpu().numpy()[best] if conf is not None else np.ones(len(xy))
        return np.column_stack([xy, conf]).astype(np.float32)


class RoboflowApiDetector:
    """Облачный API Roboflow: каждый кадр отправляется на сервер (есть лимит и кредиты).

    Для полноценной обработки используйте локальный запуск (backend "roboflow"),
    этот режим оставлен для быстрой проверки без установки inference.
    """

    provides_keypoints = False
    has_actions = True

    def __init__(self, cfg, api_key=None):
        import requests

        self.cfg = cfg
        self.key = api_key or os.getenv("ROBOFLOW_API_KEY", "")
        if not self.key:
            raise ValueError("Для Roboflow нужен API-ключ (секрет ROBOFLOW_API_KEY).")
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[0-9]+", cfg.detector_model_id):
            raise ValueError("Неверный идентификатор модели Roboflow.")
        self.session = requests.Session()
        self.calls = 0

    def detect(self, frame, frame_index=0):
        if self.calls >= self.cfg.max_api_frames:
            raise RuntimeError("Достигнут установленный лимит кадров Roboflow API.")
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Не удалось кодировать кадр.")
        self.calls += 1
        try:
            response = self.session.post(
                "https://serverless.roboflow.com/" + self.cfg.detector_model_id,
                params={"api_key": self.key, "confidence": int(self.cfg.confidence * 100), "overlap": 50},
                data=base64.b64encode(data),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=(10, 90),
            )
        except Exception:
            # текст исключения может содержать URL с ключом — не пробрасываем его
            raise RuntimeError("Нет ответа от Roboflow. Проверьте сеть.") from None
        if response.status_code != 200:
            raise RuntimeError(f"Roboflow: HTTP {response.status_code}. Проверьте ключ, доступ к модели и лимит.")
        payload = response.json()
        if isinstance(payload, list):
            payload = payload[0]
        output = []
        for p in payload.get("predictions", []):
            x, y, w, h = (float(p[k]) for k in ("x", "y", "width", "height"))
            detection = make_detection(p.get("class", ""), (x - w / 2, y - h / 2, x + w / 2, y + h / 2), p["confidence"])
            if detection is not None:
                output.append(detection)
        return output


# старое имя класса — для совместимости со старыми конфигурациями/тестами
RoboflowDetector = RoboflowApiDetector


class PlayerTracker:
    """ByteTrack поверх рамок игроков. ID трека ≠ личность игрока."""

    source = "bytetrack"

    def __init__(self, fps, id_offset=0):
        import supervision as sv

        self.sv = sv
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=max(30, round(2 * fps)),  # держим потерянный трек ~2 с
            frame_rate=max(1, round(fps)),
            minimum_consecutive_frames=2,
        )
        self.offset = id_offset

    def update(self, detections):
        players = [d for d in detections if d.kind == "player"]
        if not players:
            self.tracker.update_with_detections(self.sv.Detections.empty())
            return []
        source = self.sv.Detections(
            xyxy=np.array([d.box for d in players], np.float32),
            confidence=np.array([d.confidence for d in players], np.float32),
            class_id=np.zeros(len(players), np.int32),
            data={"action": np.array([d.action for d in players])},
        )
        result = self.tracker.update_with_detections(source)
        if result.tracker_id is None:
            return []
        return [
            Detection(tuple(map(float, box)), float(conf), "player", str(action), int(tid) + self.offset)
            for box, conf, action, tid in zip(result.xyxy, result.confidence, result.data["action"], result.tracker_id)
        ]
