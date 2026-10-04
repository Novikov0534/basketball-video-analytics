"""Единый выбор кадров для анализа.

Этот модуль импортируется и основным приложением, и отдельным процессом
Roboflow (perception_worker) в другом виртуальном окружении. Поэтому здесь
только OpenCV и NumPy — никаких тяжёлых зависимостей. Оба процесса должны
выбрать одни и те же кадры с одинаковым масштабом, иначе координаты
детекций не совпадут с кадрами основного прохода.
"""

from dataclasses import dataclass
import math

import cv2


def video_info(path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Не удалось открыть видео. Используйте MP4 (H.264).")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("У видео некорректные FPS или размер.")
    return dict(
        fps=fps,
        frames=frames,
        width=width,
        height=height,
        duration=frames / fps if frames > 0 else None,
    )


@dataclass(frozen=True)
class FramePlan:
    source_fps: float
    stride: int  # анализируем каждый stride-й кадр
    start_frame: int
    frame_limit: int | None  # сколько исходных кадров пройти (None — до конца)
    total_source_frames: int
    max_width: int

    @property
    def analysis_fps(self) -> float:
        return self.source_fps / self.stride

    @property
    def expected_frames(self) -> int:
        """Оценка числа анализируемых кадров (для прогресса)."""
        available = max(0, self.total_source_frames - self.start_frame)
        if self.frame_limit is not None:
            available = min(available, self.frame_limit)
        return max(1, math.ceil(available / self.stride))


def plan_frames(path, target_fps, start_seconds=0, max_seconds=0, max_width=1280):
    info = video_info(path)
    fps = info["fps"]
    stride = max(1, math.ceil(fps / float(target_fps) - 1e-9))
    start_frame = round(float(start_seconds) * fps)
    if info["frames"] > 0 and start_frame >= info["frames"]:
        raise ValueError("Начало находится за пределами видео.")
    limit = round(float(max_seconds) * fps) if max_seconds else None
    plan = FramePlan(fps, stride, start_frame, limit, info["frames"], int(max_width))
    return info, plan


def prepare_frame(frame, max_width):
    """Уменьшает кадр до max_width и обрезает до чётных размеров (нужно кодеку)."""
    height, width = frame.shape[:2]
    if width > max_width:
        frame = cv2.resize(frame, (max_width, round(height * max_width / width)))
    height, width = frame.shape[:2]
    return frame[: height - height % 2, : width - width % 2]


def iter_frames(path, plan: FramePlan, cancel=None):
    """Выдаёт (индекс исходного кадра, время в секундах, подготовленный кадр)."""
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, plan.start_frame)
    index = plan.start_frame
    try:
        while True:
            if plan.frame_limit is not None and index - plan.start_frame >= plan.frame_limit:
                break
            if cancel is not None and cancel.is_set():
                break
            if (index - plan.start_frame) % plan.stride:
                # grab() быстрее read(): кадр не декодируется полностью
                if not capture.grab():
                    break
                index += 1
                continue
            ok, frame = capture.read()
            if not ok:
                break
            yield index, index / plan.source_fps, prepare_frame(frame, plan.max_width)
            index += 1
    finally:
        capture.release()


def sample_frames(path, plan: FramePlan, count: int):
    """Равномерная выборка кадров по фрагменту — для обучения кластеризации команд."""
    available = plan.expected_frames
    step = max(1, available // max(1, count))
    capture = cv2.VideoCapture(str(path))
    try:
        for position in range(0, available, step):
            # индексы берём из той же сетки, что и основной проход (кратны stride)
            index = plan.start_frame + position * plan.stride
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            yield index, index / plan.source_fps, prepare_frame(frame, plan.max_width)
    finally:
        capture.release()
