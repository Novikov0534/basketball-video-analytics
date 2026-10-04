"""Распознавание номеров на майках локальным классификатором ResNet.

По умолчанию веса загружаются из ``models/jersey_resnet.pt``. Модель
классифицирует найденный детектором кроп номера среди обученных классов.
"""

from pathlib import Path

import cv2
import numpy as np

IMAGE_SIZE = 96
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare_crop(crop, size=IMAGE_SIZE):
    """Кроп номера → тензорное представление: квадрат с паддингом, нормировка.

    Номера бывают вытянутыми (одна цифра) и широкими (две), поэтому кроп
    вписывается в квадрат с сохранением пропорций, а не растягивается.
    """
    if crop is None or crop.size == 0:
        return None
    height, width = crop.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((size, size, 3), np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((image - MEAN) / STD).transpose(2, 0, 1)


class JerseyNumberClassifier:
    """ResNet-классификатор номера по кропу майки."""

    name = "resnet"

    def __init__(self, weights, device="auto", batch_size=64, min_confidence=0.55):
        import torch

        self.torch = torch
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        self.classes = list(checkpoint["classes"])
        self.architecture = checkpoint.get("architecture", "resnet34")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.model = build_model(self.architecture, len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

    def predict(self, crops):
        """→ [(номер, уверенность)]; номер пустой, если модель не уверена."""
        prepared = [prepare_crop(crop) for crop in crops]
        valid = [i for i, item in enumerate(prepared) if item is not None]
        results = [("", 0.0)] * len(crops)
        with self.torch.no_grad():
            for start in range(0, len(valid), self.batch_size):
                chunk = valid[start:start + self.batch_size]
                batch = self.torch.from_numpy(np.stack([prepared[i] for i in chunk])).to(self.device)
                probabilities = self.torch.softmax(self.model(batch), dim=1)
                confidence, index = probabilities.max(dim=1)
                for position, class_index, score in zip(chunk, index.tolist(), confidence.tolist()):
                    label = self.classes[class_index]
                    results[position] = (label if score >= self.min_confidence else "", round(score, 3))
        return results


def build_model(architecture, class_count):
    """ResNet из torchvision без предобученных весов (их загружает обучение)."""
    from torchvision import models

    factory = getattr(models, architecture, None)
    if factory is None:
        raise ValueError(f"Неизвестная архитектура: {architecture}")
    model = factory(weights=None)
    model.fc = _linear(model.fc.in_features, class_count)
    return model


def _linear(in_features, out_features):
    from torch import nn

    return nn.Linear(in_features, out_features)


def default_weights(root=None):
    """Путь к обученным весам, если они есть."""
    folder = Path(root or Path(__file__).resolve().parents[1] / "models")
    candidate = folder / "jersey_resnet.pt"
    return candidate if candidate.is_file() else None
