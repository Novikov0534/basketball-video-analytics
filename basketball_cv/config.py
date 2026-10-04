"""Параметры запуска Basketball CV.

Основной пользовательский режим полностью автоматический. Ручная калибровка,
ROI табло и перестановка команд сохранены только для CLI/оценки качества и не
показываются в основном веб-интерфейсе.
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import math

from .court import COURTS

BACKENDS = ("roboflow", "roboflow_api", "yolo", "replay")


@dataclass
class Config:
    # --- детектор ---------------------------------------------------------
    backend: str = "roboflow"  # roboflow (локально) | roboflow_api | yolo | replay (демо)
    detector_model_id: str = "basketball-player-detection-3-ycjdo/13"
    keypoint_model_id: str = "basketball-court-detection-2/14"
    number_model_id: str = "basketball-jersey-numbers-ocr/3"
    number_reader: str = "auto"  # ResNet при наличии весов, иначе SmolVLM2
    number_weights: str = ""  # свои веса ResNet; пусто — models/jersey_resnet.pt
    read_numbers: bool = True
    number_every: int = 5  # устаревшее: чтение подряд (оставлено для старых конфигураций)
    keypoint_every: int = 3  # искать ключевые точки площадки на каждом N-м кадре
    number_crops_per_track: int = 8  # сколько лучших и разнесённых по времени кропов читать на игрока
    weights: str = "yolo11n.pt"  # для backend=yolo: COCO-базовая линия или свои веса
    court_keypoint_weights: str = ""  # для backend=yolo: свои веса YOLO-pose площадки
    device: str = "auto"
    tracker: str = "bytetrack"  # UI выбирает SAM2; CLI остаётся совместим с любым backend
    tracker_fallback: bool = True  # если SAM2 не запустился, продолжить на ByteTrack
    image_size: int = 960
    confidence: float = 0.4
    max_api_frames: int = 600
    # --- фрагмент видео ---------------------------------------------------
    target_fps: float = 30  # FPS тяжёлого анализа
    output_fps: float = 30  # FPS итогового видео (не выше исходного)
    start_seconds: float = 0
    max_seconds: float = 0  # 0 — до конца
    max_width: int = 1280
    # --- площадка ----------------------------------------------------------
    league: str = "nba"  # nba | fiba
    calibration: str = "auto"  # manual/none оставлены только для CLI и evaluation
    image_points: list = field(default_factory=list)  # ручная калибровка, доли кадра [0..1]
    court_points: list = field(default_factory=list)  # те же точки на площадке, м
    camera_mode: str = "moving"  # для ручной калибровки: fixed | moving
    hoops: list = field(default_factory=list)  # ручные области колец (если детектор их не видит)
    # --- команды и игроки --------------------------------------------------
    team_names: list = field(default_factory=lambda: ["Команда 1", "Команда 2"])
    rosters: list = field(default_factory=lambda: [{}, {}])  # номер → имя, для каждой команды
    team_embedder: str = "auto"  # auto (SigLIP + цвет) | siglip | color
    team_reducer: str = "pca"  # pca (основной) | umap (альтернативный)
    swap_teams: bool = False  # служебный параметр CLI/evaluation; в UI не показывается
    # --- табло (необязательно: только для сверки со счётом по броскам) -----
    score_rois: dict = field(default_factory=dict)
    initial_score: list | None = None
    ocr_interval: float = 0.5
    # --- измерения ---------------------------------------------------------
    max_speed_m_s: float = 12
    metrics_interval: float = 0.5
    smoothing_window: int = 5
    dead_zone_m: float = 0.08

    @property
    def court(self):
        return COURTS[self.league]

    def tune_number_rate(self, seconds):
        """Выбирает число лучших кропов номера на один трек.

        И ResNet, и SmolVLM читают только несколько крупных кадров с разными
        ракурсами. Это стабилизирует голосование и не заставляет классификатор
        повторно обрабатывать сотни почти одинаковых изображений одной майки.
        """
        if not seconds:
            self.number_crops_per_track = 10
        elif seconds > 120:
            self.number_crops_per_track = 6
        elif seconds > 30:
            self.number_crops_per_track = 8
        else:
            self.number_crops_per_track = 10
        return self

    def validate(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"Неизвестный детектор: {self.backend}.")
        if self.league not in COURTS:
            raise ValueError("Стандарт площадки: nba или fiba.")
        if self.tracker not in ("bytetrack", "sam2"):
            raise ValueError("Трекер: bytetrack или sam2.")
        if self.number_reader not in ("auto", "smolvlm", "resnet"):
            raise ValueError("Чтение номеров: auto, smolvlm или resnet.")
        if self.team_reducer not in ("pca", "umap"):
            raise ValueError("Понижение размерности: pca или umap.")
        if self.tracker == "sam2" and self.backend not in ("roboflow", "replay"):
            raise ValueError("SAM2 доступен только с локальными моделями Roboflow (backend=roboflow).")
        if self.calibration not in ("auto", "manual", "none"):
            raise ValueError("Калибровка: auto, manual или none.")
        if self.camera_mode not in ("fixed", "moving"):
            raise ValueError("Неизвестный режим камеры.")
        for key in ("target_fps", "output_fps", "max_speed_m_s", "metrics_interval", "ocr_interval"):
            value = float(getattr(self, key))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key}: требуется положительное конечное число.")
        if not 1 <= self.target_fps <= 60 or not 1 <= self.output_fps <= 60 or not 0.01 <= self.confidence <= 1:
            raise ValueError("Недопустимый FPS / порог уверенности.")
        if any(not math.isfinite(v) or v < 0 for v in (self.start_seconds, self.max_seconds)):
            raise ValueError("Недопустимый интервал видео.")
        if not 320 <= self.image_size <= 1536 or not 320 <= self.max_width <= 3840:
            raise ValueError("Недопустимый размер изображения.")
        if self.keypoint_every < 1 or self.number_crops_per_track < 1:
            raise ValueError("Недопустимые параметры разрежения.")
        if self.max_api_frames < 1 or self.number_every < 1 or self.smoothing_window < 1:
            raise ValueError("Недопустимые лимиты.")
        if len(self.image_points) != len(self.court_points) or (self.image_points and len(self.image_points) < 4):
            raise ValueError("Для ручной калибровки нужны минимум 4 пары точек.")
        for x, y in self.image_points:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("Координаты изображения должны лежать в [0,1].")
        for xy in self.court_points:
            if not self.court.contains(xy, margin=0.01):
                raise ValueError("Координаты площадки задаются в метрах внутри её границ.")
        for roi in list(self.score_rois.values()) + list(self.hoops):
            if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
                raise ValueError("Некорректная прямоугольная область.")
        if len(self.team_names) != 2 or len(self.rosters) != 2:
            raise ValueError("Нужно указать две команды.")
        if self.initial_score is not None and (
            len(self.initial_score) != 2 or any(int(v) != v or v < 0 for v in self.initial_score)
        ):
            raise ValueError("Счёт — два неотрицательных целых числа.")
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        # совместимость с конфигурациями версии 0.1
        if "roboflow_model_id" in data:
            data.setdefault("detector_model_id", data.pop("roboflow_model_id"))
        if data.get("court_length") and abs(float(data["court_length"]) - 28.65) < 0.05:
            data.setdefault("league", "nba")
        if data.get("image_points") and "calibration" not in data:
            data["calibration"] = "manual"
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8"))).validate()


