"""Кэш восприятия и запуск процесса Roboflow.

Тяжёлая часть (нейросети) выполняется один раз и сохраняется в JSONL.
Аналитику (владение, события, статистику) можно пересчитывать сколько угодно
раз без повторного инференса — достаточно тех же параметров кадров.
"""

from pathlib import Path
import bisect
import hashlib
import json
import os
import subprocess
import sys
import uuid

import numpy as np

from .detectors import make_detection

ROOT = Path(__file__).resolve().parents[1]


class CachedPerception:
    """Детекции и ключевые точки площадки из файла JSONL.

    Реализует тот же интерфейс, что и живые детекторы: detect(frame, index)
    и keypoints(frame, index). Используется и для Roboflow, и для демо.
    """

    provides_keypoints = True
    has_actions = True
    provides_tracks = False

    def __init__(self, path):
        self.path = Path(path)
        self.header, self.frames, complete = {}, {}, False
        with self.path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                kind = record.get("type")
                if kind == "header":
                    self.header = record
                elif kind == "frame":
                    self.frames[int(record["frame"])] = record
                elif kind == "end":
                    complete = True
        if not complete:
            raise ValueError(f"Файл восприятия не завершён: {self.path.name}")
        self._keypoint_frames = sorted(i for i, r in self.frames.items() if r.get("keypoints"))
        labels = {d.get("label", "") for r in self.frames.values() for d in r.get("detections", [])}
        # кэш несёт готовые треки, только если восприятие вело их само (SAM2)
        self.tracker = str(self.header.get("tracker", "none"))
        self.provides_tracks = any(d.get("track") is not None
                                   for r in self.frames.values() for d in r.get("detections", []))
        self.has_actions = any("jump-shot" in l or "ball-in-basket" in l for l in labels)
        self.provides_keypoints = bool(self._keypoint_frames)

    def detect(self, frame=None, frame_index=0):
        record = self.frames.get(int(frame_index))
        if record is None:
            return []
        output = []
        for d in record.get("detections", []):
            detection = make_detection(d.get("label", ""), d["box"], d.get("conf", 0),
                                       d.get("text", ""), d.get("track"))
            if detection is not None:
                output.append(detection)
        return output

    def keypoints(self, frame=None, frame_index=0):
        """Ключевые точки кадра; между опорными кадрами — линейная интерполяция.

        Модель площадки запускается не на каждом кадре (это экономит время),
        но камера движется плавно, поэтому промежуточные положения точек
        восстанавливаются интерполяцией без потери качества гомографии.
        """
        index = int(frame_index)
        record = self.frames.get(index)
        if record and record.get("keypoints"):
            return np.asarray(record["keypoints"], dtype=np.float32)
        if not self._keypoint_frames:
            return None
        position = bisect.bisect_left(self._keypoint_frames, index)
        before = self._keypoint_frames[position - 1] if position else None
        after = self._keypoint_frames[position] if position < len(self._keypoint_frames) else None
        if before is None or after is None:
            nearest = before if after is None else after
            return np.asarray(self.frames[nearest]["keypoints"], dtype=np.float32)
        alpha = (index - before) / (after - before)
        first = np.asarray(self.frames[before]["keypoints"], dtype=np.float32)
        second = np.asarray(self.frames[after]["keypoints"], dtype=np.float32)
        mixed = first * (1 - alpha) + second * alpha
        # точка считается надёжной, только если уверенно найдена в обоих кадрах
        mixed[:, 2] = np.minimum(first[:, 2], second[:, 2])
        return mixed


def worker_python():
    """Python окружения .venv-rf (создаётся scripts/bootstrap.py --roboflow)."""
    explicit = os.getenv("BCV_ROBOFLOW_PYTHON")
    if explicit:
        return Path(explicit)
    folder = ROOT / ".venv-rf"
    return folder / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def cache_key(video, cfg):
    """Ключ кэша: видео + всё, что влияет на выбор кадров и модели."""
    stat = Path(video).stat()
    parts = [
        Path(video).name, stat.st_size, int(stat.st_mtime),
        cfg.target_fps, cfg.start_seconds, cfg.max_seconds, cfg.max_width,
        cfg.detector_model_id, cfg.keypoint_model_id, cfg.confidence, cfg.keypoint_every, cfg.tracker,
    ]
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def run_roboflow_worker(video, cfg, cache_dir, api_key=None, progress=None, cancel=None):
    """Запускает perception_worker в окружении .venv-rf; возвращает путь к JSONL.

    Если такой файл уже есть (те же видео и параметры), инференс не повторяется.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"perception-{cache_key(video, cfg)}.jsonl"
    if target.exists():
        if progress:
            progress(1.0, "Восприятие взято из кэша (повторный инференс не нужен)")
        return target
    python = worker_python()
    if not python.exists():
        raise RuntimeError(
            "Не установлено окружение Roboflow (.venv-rf). Выполните: "
            "python scripts/bootstrap.py --roboflow (в Colab это делает ячейка установки)."
        )
    command = [
        str(python), "-u", str(ROOT / "basketball_cv" / "perception_worker.py"),
        "--video", str(video), "--out", str(target),
        "--target-fps", str(cfg.target_fps), "--start", str(cfg.start_seconds),
        "--max-seconds", str(cfg.max_seconds), "--max-width", str(cfg.max_width),
        "--detector-model", cfg.detector_model_id,
        "--keypoint-model", cfg.keypoint_model_id,
        "--keypoint-every", str(cfg.keypoint_every),
        "--number-model", "",  # номера читаются отдельным проходом по выбранным кропам
        "--confidence", str(cfg.confidence),
    ]
    if cfg.tracker == "sam2":
        checkpoint, config = sam2_files()
        if checkpoint is None:
            if not cfg.tracker_fallback:
                raise RuntimeError("SAM2 не установлен: выполните scripts/bootstrap.py --roboflow --sam2.")
            if progress:
                progress(0.0, "SAM2 не установлен — отслеживание выполнит ByteTrack")
        else:
            command += ["--tracker", "sam2", "--sam2-checkpoint", str(checkpoint), "--sam2-config", config]
    environment = dict(os.environ)
    if api_key:
        environment["ROBOFLOW_API_KEY"] = api_key
    if not environment.get("ROBOFLOW_API_KEY"):
        raise ValueError("Нужен ROBOFLOW_API_KEY: секрет Colab, переменная окружения или поле в интерфейсе.")
    tail = []
    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True, bufsize=1, env=environment) as process:
        for line in process.stderr:
            if cancel is not None and cancel.is_set():
                process.terminate()
                break
            if line.startswith("PROGRESS"):
                _, done, total, speed = line.split()
                if progress:
                    progress(min(1.0, int(done) / max(1, int(total))),
                             f"Нейросети Roboflow: кадр {done}/{total} · {float(speed):.1f} кадр/с")
            else:
                tail = (tail + [line.rstrip()])[-25:]
        code = process.wait()
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Обработка остановлена на этапе нейросетей.")
    if code != 0 or not target.exists():
        details = "\n".join(line for line in tail if line)
        secret = environment.get("ROBOFLOW_API_KEY", "")
        if secret:
            details = details.replace(secret, "***")  # ключ не должен попасть в интерфейс
        raise RuntimeError(f"Процесс Roboflow завершился с ошибкой (код {code}).\n{details}")
    return target


def read_number_crops(video, cfg, requests, api_key=None, progress=None, cancel=None, work_dir=None):
    """Читает номера только на выбранных рамках (см. identity.select_number_crops).

    Возвращает [(track_id, текст)]. Это второй, дешёвый вызов моделей Roboflow:
    вместо тысяч рамок читается несколько десятков лучших кропов на игрока.
    """
    if not requests:
        return []
    work_dir = Path(work_dir or (ROOT / "outputs" / "_cache"))
    work_dir.mkdir(parents=True, exist_ok=True)
    crops_file = work_dir / f"crops-{uuid.uuid4().hex[:8]}.json"
    crops_out = crops_file.with_name(crops_file.stem + "-read.jsonl")
    crops_file.write_text(json.dumps(requests), encoding="utf-8")
    python = worker_python()
    if not python.exists():
        return []
    command = [
        str(python), "-u", str(ROOT / "basketball_cv" / "perception_worker.py"),
        "--video", str(video), "--out", str(crops_out),
        "--target-fps", str(cfg.target_fps), "--start", str(cfg.start_seconds),
        "--max-seconds", str(cfg.max_seconds), "--max-width", str(cfg.max_width),
        "--number-model", cfg.number_model_id,
        "--crops", str(crops_file), "--crops-out", str(crops_out),
    ]
    environment = dict(os.environ, MPLBACKEND="Agg")
    if api_key:
        environment["ROBOFLOW_API_KEY"] = api_key
    try:
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, bufsize=1, env=environment) as process:
            for line in process.stderr:
                if cancel is not None and cancel.is_set():
                    process.terminate()
                    break
                if line.startswith("PROGRESS") and progress:
                    _, done, total, _ = line.split()
                    progress(int(done) / max(1, int(total)), f"Чтение номеров: {done}/{total} кропов")
            process.wait()
        if not crops_out.exists():
            return []
        results = []
        for line in crops_out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("text"):
                    results.append((int(record["track_id"]), record["text"]))
        return results
    finally:
        crops_file.unlink(missing_ok=True)
        crops_out.unlink(missing_ok=True)


SAM2_CHECKPOINTS = (
    ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
)


def sam2_files(root=None):
    """→ (путь к весам, имя конфигурации) или (None, None), если SAM2 не установлен.

    Достаточно наличия весов: сам код SAM2 работает и без сборки пакета,
    воркер подхватит его из папки с исходниками.
    """
    folder = Path(root or (ROOT / ".sam2" / "checkpoints"))
    for name, config in SAM2_CHECKPOINTS:
        candidate = folder / name
        if candidate.is_file():
            return candidate, config
    return None, None
