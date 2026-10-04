"""Процесс восприятия на моделях Roboflow (запускается в отдельном окружении .venv-rf).

Зачем отдельный процесс: пакет `inference-gpu` тянет ~230 зависимостей
(свой torch, supervision 0.29, pillow 12, onnxruntime-gpu...), которые
конфликтуют с закреплёнными версиями основного приложения. Поэтому модели
Roboflow работают здесь, а результат передаётся через файл JSONL:

    {"type": "header", ...параметры...}
    {"type": "frame", "frame": 123, "time": 4.1,
     "detections": [{"label": "player", "box": [x1,y1,x2,y2], "conf": 0.9, "track": 3}, ...],
     "keypoints": [[x, y, conf], ... 33 шт.] | null}
    ...
    {"type": "end", "frames": N}

Модели (те же, что в ноутбуке Roboflow):
    детектор  basketball-player-detection-3-ycjdo/13 (RF-DETR Medium, 10 классов);
    площадка  basketball-court-detection-2/14 (33 ключевые точки);
    номера    basketball-jersey-numbers-ocr/3 (SmolVLM2 + LoRA).

Отслеживание: без специального режима треки ведёт ByteTrack в основном приложении.
С ключом --tracker sam2 здесь же работает SAM2: рамки игроков с опорного кадра подаются как промпты, дальше модель
сегментирует и ведёт каждого игрока сама, а поле "track" переносит её
идентификаторы в кэш.

Модуль нарочно не импортирует ничего из основного приложения, кроме frames.py.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

NUMBER_PROMPT = "Read the number."
NUMBER_PATTERN = re.compile(r"^\d{1,2}$")


def as_dict(result):
    """Ответ inference (pydantic-модель) → обычный dict."""
    if isinstance(result, dict):
        return result
    for method in ("model_dump", "dict"):
        if hasattr(result, method):
            try:
                return getattr(result, method)(by_alias=True, exclude_none=True)
            except TypeError:
                return getattr(result, method)()
    raise TypeError(f"Неизвестный формат ответа модели: {type(result)!r}")


def parse_detections(result):
    """Ответ детектора → [{"label", "box", "conf"}] в пикселях."""
    output = []
    for p in as_dict(result).get("predictions", []):
        x, y, w, h = (float(p[k]) for k in ("x", "y", "width", "height"))
        output.append(
            {
                "label": str(p.get("class", p.get("class_name", ""))),
                "box": [round(x - w / 2, 2), round(y - h / 2, 2), round(x + w / 2, 2), round(y + h / 2, 2)],
                "conf": round(float(p.get("confidence", 0)), 4),
            }
        )
    return output


def parse_keypoints(result, count=33):
    """Ответ модели ключевых точек → список из `count` троек [x, y, conf].

    Если найдено несколько «площадок», берём с наибольшей уверенностью.
    Точки раскладываются по class_id (номер точки в скелете), а при его
    отсутствии — по порядку.
    """
    predictions = as_dict(result).get("predictions", [])
    if not predictions:
        return None
    best = max(predictions, key=lambda p: float(p.get("confidence", 0)))
    points = [[0.0, 0.0, 0.0] for _ in range(count)]
    for order, kp in enumerate(best.get("keypoints", [])):
        index = kp.get("class_id", order)
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = order
        if 0 <= index < count:
            points[index] = [round(float(kp["x"]), 2), round(float(kp["y"]), 2), round(float(kp.get("confidence", 0)), 4)]
    return points


def read_number(model, crop):
    """Номер на майке через SmolVLM2. Поддерживает API inference 0.x и 1.x."""
    text = None
    try:
        response = model.infer(crop, prompt=NUMBER_PROMPT)
        first = response[0] if isinstance(response, (list, tuple)) else response
        text = getattr(first, "response", None)
        if text is None:
            text = as_dict(first).get("response")
    except Exception:
        text = None
    if text is None:
        try:
            text = model.predict(crop, NUMBER_PROMPT)[0]
        except Exception:
            return ""
    text = str(text).strip()
    return text if NUMBER_PATTERN.fullmatch(text) else ""


def crop_box(frame, box, pad=10):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
    x2, y2 = min(width, int(x2 + pad)), min(height, int(y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def process_frame(frame, models, options, position):
    """Один кадр → запись JSONL. models: dict с ключами detector/keypoints/numbers."""
    record = {"detections": [], "keypoints": None}
    result = models["detector"].infer(frame, confidence=options.confidence, iou_threshold=options.iou)[0]
    record["detections"] = parse_detections(result)
    # площадка меняется плавно, поэтому ключевые точки ищем не в каждом кадре;
    # промежуточные значения приложение интерполирует само
    if models.get("keypoints") is not None and position % max(1, options.keypoint_every) == 0:
        kp_result = models["keypoints"].infer(frame, confidence=options.keypoint_confidence)[0]
        record["keypoints"] = parse_keypoints(kp_result)
    if models.get("numbers") is not None and position % max(1, options.number_every) == 0:
        for detection in record["detections"]:
            if detection["label"] == "number":
                crop = crop_box(frame, detection["box"])
                if crop is not None:
                    detection["text"] = read_number(models["numbers"], crop)
    return record


def read_crops(options):
    """Режим чтения номеров по заранее выбранным рамкам.

    Приложение сначала прогоняет детектор и трекер, выбирает по несколько
    лучших кропов на каждого игрока и только потом просит прочитать их.
    Так вместо тысяч вызовов SmolVLM получается сотня-полторы.
    """
    from basketball_cv.frames import iter_frames, plan_frames

    requests = json.loads(Path(options.crops).read_text(encoding="utf-8"))
    wanted = {}
    for item in requests:
        wanted.setdefault(int(item["frame"]), []).append(item)
    _, plan = plan_frames(options.video, options.target_fps, options.start, options.max_seconds, options.max_width)
    model = load_models(options, only="numbers")["numbers"]
    done = 0
    with Path(options.crops_out).with_suffix(".partial").open("w", encoding="utf-8") as sink:
        for index, _, frame in iter_frames(options.video, plan):
            for item in wanted.get(index, []):
                crop = crop_box(frame, item["box"])
                text = read_number(model, crop) if crop is not None else ""
                sink.write(json.dumps(dict(track_id=item["track_id"], frame=index, text=text)) + "\n")
                done += 1
                print(f"PROGRESS {done} {len(requests)} 0", file=sys.stderr, flush=True)
    Path(options.crops_out).with_suffix(".partial").replace(Path(options.crops_out))
    print(f"DONE {done}", file=sys.stderr, flush=True)


def add_sam2_to_path(checkpoint):
    """Даёт импортировать SAM2 из папки с исходниками, если пакет не установлен.

    Сборка пакета требует совпадения версий CUDA и компилятора и удаётся не
    всегда, а код на Python работает и без установки. Папка с исходниками
    ставится в начало пути поиска: в окружении может оказаться официальный
    пакет SAM2 от Meta, у которого нет потокового интерфейса.
    """
    root = Path(checkpoint).resolve().parent.parent  # .sam2/checkpoints/веса.pt → .sam2
    if not (root / "sam2").is_dir():
        return
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    for name in [key for key in sys.modules if key == "sam2" or key.startswith("sam2.")]:
        del sys.modules[name]  # переимпортируем именно из исходников


def load_sam2_builders():
    """→ (потоковый построитель или None, видеопостроитель или None, откуда взят).

    Есть две реализации SAM2: форк с потоковым интерфейсом
    (`build_sam2_camera_predictor`, кадр за кадром — как в ноутбуке Roboflow) и
    официальный пакет Meta, который умеет только обрабатывать видео целиком
    (`build_sam2_video_predictor`). Поддерживаем обе.
    """
    import sam2.build_sam as builders

    return (getattr(builders, "build_sam2_camera_predictor", None),
            getattr(builders, "build_sam2_video_predictor", None),
            getattr(builders, "__file__", "?"))


class Sam2Tracker:
    """Потоковое отслеживание SAM2 (форк с интерфейсом камеры, как в ноутбуке Roboflow).

    Рамки детектора подаются как промпты на опорном кадре, дальше SAM2 ведёт
    каждого игрока по кадрам своим механизмом памяти и переживает перекрытия.
    Отличие от оригинала: промпты подаются заново после монтажной склейки и
    когда часть игроков потеряна, поэтому ролик не обязан начинаться кадром,
    где видны все.
    """

    streaming = True

    def __init__(self, checkpoint, config, device="cuda"):
        import torch

        add_sam2_to_path(checkpoint)
        build_sam2_camera_predictor, _, source = load_sam2_builders()
        if build_sam2_camera_predictor is None:
            raise RuntimeError(
                f"Установленный SAM2 ({source}) без потокового интерфейса — "
                "используйте видеорежим (Sam2VideoTracker)."
            )
        self.torch = torch
        self.predictor = build_sam2_camera_predictor(config, checkpoint, device=device)
        self.device = device
        self.dtype = pick_dtype(torch, device)
        self.next_id = 1
        self.prompted = False

    def _autocast(self):
        if self.dtype is None:
            return self.torch.autocast("cpu", enabled=False)
        return self.torch.autocast("cuda", dtype=self.dtype)

    def prompt(self, frame, boxes, previous=()):
        """Задать объекты для отслеживания по рамкам детектора.

        При повторной подаче промптов идентификаторы сохраняются: новая рамка
        наследует номер того трека, с которым она сильнее всего пересекается.
        Без этого каждая переподача начинала бы игроков заново, и трекинг рвался
        бы чаще, чем без SAM2 вообще.
        """
        import numpy as np

        if not len(boxes):
            return []
        ids = self._inherit_ids(boxes, previous)
        with self.torch.inference_mode(), self._autocast():
            self.predictor.load_first_frame(frame)
            for box, object_id in zip(boxes, ids):
                self.predictor.add_new_prompt(frame_idx=0, obj_id=int(object_id),
                                              bbox=np.asarray([box], dtype=np.float32))
        self.prompted = True
        return ids

    def _inherit_ids(self, boxes, previous, iou_threshold=0.3):
        pairs = sorted(
            ((box_iou(box, old_box), index, old_id)
             for index, box in enumerate(boxes)
             for old_id, old_box in previous
             if box_iou(box, old_box) >= iou_threshold),
            reverse=True,
        )
        ids = [None] * len(boxes)
        taken = set()
        for _, index, old_id in pairs:
            if ids[index] is None and old_id not in taken:
                ids[index] = old_id
                taken.add(old_id)
        for index, value in enumerate(ids):
            if value is None:
                ids[index] = self.next_id
                self.next_id += 1
        return ids

    def track(self, frame):
        """→ [(track_id, [x1, y1, x2, y2])] для текущего кадра."""
        import numpy as np

        if not self.prompted:
            return []
        with self.torch.inference_mode(), self._autocast():
            ids, logits = self.predictor.track(frame)
        masks = np.squeeze((logits > 0.0).cpu().numpy()).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        output = []
        for object_id, mask in zip(np.asarray(ids).ravel().tolist(), masks):
            box = mask_to_box(clean_mask(mask))
            if box is not None:
                output.append((int(object_id), box))
        return drop_duplicate_tracks(output)


class Sam2VideoTracker:
    """Отслеживание официальным SAM2 от Meta: видео обрабатывается целиком.

    У официального пакета нет потокового интерфейса: он принимает папку с
    кадрами, получает промпты на опорных кадрах и за один проход выдаёт маски
    для всего ролика. Поэтому работа идёт в два этапа — сначала детектор
    по кадрам, потом отслеживание, — а не покадрово, как в форке.

    Промпты подаются на нескольких опорных кадрах, чтобы игроки, появившиеся
    позже начала ролика, тоже попадали в отслеживание.
    """

    streaming = False

    def __init__(self, checkpoint, config, device="cuda"):
        import torch

        add_sam2_to_path(checkpoint)
        _, build_sam2_video_predictor, source = load_sam2_builders()
        if build_sam2_video_predictor is None:
            raise RuntimeError(f"В установленном SAM2 ({source}) нет ни одного известного интерфейса.")
        self.torch = torch
        self.predictor = build_sam2_video_predictor(config, checkpoint, device=device)
        self.device = device
        self.dtype = pick_dtype(torch, device)

    def _autocast(self):
        if self.dtype is None:
            return self.torch.autocast("cpu", enabled=False)
        return self.torch.autocast("cuda", dtype=self.dtype)

    def run(self, frames_dir, player_boxes, frame_count, rounds=3, coverage=0.75):
        """Отслеживание по всему фрагменту. → {позиция кадра: [(track_id, рамка)]}.

        Промпты подаются итеративно, а не «каждые N секунд». Сначала модель
        получает игроков с первого кадра и проходит весь ролик. Потом мы
        смотрим, где детектор видит игроков, а трека для них нет: там
        добавляем недостающих и проходим ещё раз.

        Так сделано потому, что подавать промпты по расписанию нельзя: за
        пару секунд игрок перемещается, его рамка перестаёт пересекаться с
        прежней, и он выглядит «новым». Из-за этого на каждой переподаче
        появлялись дубликаты — по лишнему треку на игрока.
        """
        import numpy as np

        with self.torch.inference_mode(), self._autocast():
            state = self.predictor.init_state(video_path=str(frames_dir))
            next_id = 1
            for box in player_boxes.get(0, []):  # первый кадр: все, кого видит детектор
                self.predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=next_id, box=np.asarray(box, dtype=np.float32))
                next_id += 1
            if next_id == 1:
                return {}
            tracks = self._propagate(state)
            for _ in range(max(0, rounds - 1)):
                position, missing = self._find_uncovered(player_boxes, tracks, coverage)
                if position is None:
                    break
                print(f"SAM2: добавляю {len(missing)} игроков на кадре {position}",
                      file=sys.stderr, flush=True)
                for box in missing:
                    self.predictor.add_new_points_or_box(
                        state, frame_idx=int(position), obj_id=next_id,
                        box=np.asarray(box, dtype=np.float32))
                    next_id += 1
                tracks = self._propagate(state)
        return tracks

    def _propagate(self, state):
        import numpy as np

        output = {}
        for position, object_ids, logits in self.predictor.propagate_in_video(state):
            masks = (logits > 0.0).cpu().numpy()
            boxes = []
            for object_id, mask in zip(object_ids, masks):
                box = mask_to_box(clean_mask(np.squeeze(mask).astype(bool)))
                if box is not None:
                    boxes.append((int(object_id), box))
            output[int(position)] = drop_duplicate_tracks(boxes)
        return output

    @staticmethod
    def _find_uncovered(player_boxes, tracks, coverage, iou_threshold=0.3):
        """Кадр, где отслеживается меньше всего игроков. → (позиция, недостающие рамки).

        Возвращает первый кадр, в котором доля покрытых треками детекций ниже
        порога: именно там кто-то вышел на площадку или был потерян.
        """
        for position in sorted(player_boxes):
            boxes = player_boxes[position]
            if not boxes:
                continue
            existing = [box for _, box in tracks.get(position, [])]
            missing = [box for box in boxes
                       if all(box_iou(box, other) < iou_threshold for other in existing)]
            if missing and len(existing) < coverage * len(boxes):
                return position, missing
        return None, []


def drop_duplicate_tracks(boxes, iou_threshold=0.75):
    """Убирает почти совпадающие рамки: один игрок — один трек на кадре.

    SAM2 может вести двух «объектов» по одному человеку, если он получил
    промпты дважды. Оставляем трек с меньшим номером — он появился раньше.
    """
    kept = []
    for track_id, box in sorted(boxes):
        if all(box_iou(box, other) < iou_threshold for _, other in kept):
            kept.append((track_id, box))
    return kept


def pick_dtype(torch, device):
    """bfloat16 на Ampere и новее, float16 на Turing (T4), ничего на процессоре."""
    if device != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def clean_mask(mask, relative_distance=0.03):
    """Убирает от маски куски, оторванные от основного тела игрока.

    SAM2 в динамичных сценах иногда прихватывает мяч или соседа: маска
    распадается на несколько частей. Оставляем самую крупную и всё, что рядом
    с ней, остальное отбрасываем.
    """
    import cv2
    import numpy as np

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = int(np.argmax(areas)) + 1
    limit = relative_distance * float(np.hypot(*mask.shape))
    keep = np.zeros_like(mask)
    for label in range(1, count):
        if label == main or np.linalg.norm(centroids[label] - centroids[main]) <= limit:
            keep |= labels == label
    return keep


def mask_to_box(mask):
    import numpy as np

    rows, columns = np.where(mask)
    if rows.size == 0:
        return None
    return [float(columns.min()), float(rows.min()), float(columns.max()), float(rows.max())]


def box_iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    overlap = width * height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
    return overlap / union if union > 0 else 0.0


def is_player(label):
    return str(label).startswith("player") or str(label) == "person"


def attach_tracks(detections, tracks, iou_threshold=0.3):
    """Переносит идентификаторы SAM2 на детекции игроков по наибольшему IoU.

    Рамки берём от детектора (они точнее по краям), идентификаторы — от SAM2.
    Трек, которому не нашлось детекции, добавляется отдельной записью: игрок
    виден, просто на этом кадре детектор его пропустил.
    """
    players = [d for d in detections if is_player(d["label"])]
    used_tracks, used_players = set(), set()
    pairs = []
    for track_index, (track_id, box) in enumerate(tracks):
        for player_index, detection in enumerate(players):
            score = box_iou(box, detection["box"])
            if score >= iou_threshold:
                pairs.append((score, track_index, player_index))
    for score, track_index, player_index in sorted(pairs, reverse=True):
        if track_index in used_tracks or player_index in used_players:
            continue
        used_tracks.add(track_index)
        used_players.add(player_index)
        players[player_index]["track"] = tracks[track_index][0]
    for track_index, (track_id, box) in enumerate(tracks):
        if track_index not in used_tracks:
            detections.append(dict(label="player", box=[round(v, 2) for v in box],
                                   conf=0.3, track=track_id, from_tracker=True))
    return len(used_tracks), len(players)


def build_tracker(options):
    """Трекер SAM2 в доступной реализации: потоковой (форк) или видеорежимной."""
    if options.tracker != "sam2":
        return None
    device = sam2_device()
    try:
        tracker = Sam2Tracker(options.sam2_checkpoint, options.sam2_config, device)
        print(f"SAM2 потоковый: {options.sam2_checkpoint}, точность {tracker.dtype}",
              file=sys.stderr, flush=True)
        return tracker
    except RuntimeError as exc:
        print(f"Потоковый интерфейс SAM2 недоступен ({exc}); перехожу в видеорежим",
              file=sys.stderr, flush=True)
    tracker = Sam2VideoTracker(options.sam2_checkpoint, options.sam2_config, device)
    print(f"SAM2 видеорежим: {options.sam2_checkpoint}, точность {tracker.dtype}",
          file=sys.stderr, flush=True)
    return tracker


def sam2_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models(options, only=None):
    os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CUDAExecutionProvider]")
    from inference import get_model

    key = os.getenv("ROBOFLOW_API_KEY", "")
    if not key:
        raise SystemExit("Нет ROBOFLOW_API_KEY: добавьте секрет в Colab или переменную окружения.")
    if only == "numbers":
        return {"numbers": get_model(model_id=options.number_model, api_key=key)}
    if only == "detector":
        return {"detector": get_model(model_id=options.detector_model, api_key=key)}
    models = {"detector": get_model(model_id=options.detector_model, api_key=key)}
    models["keypoints"] = get_model(model_id=options.keypoint_model, api_key=key) if options.keypoint_model else None
    models["numbers"] = get_model(model_id=options.number_model, api_key=key) if options.number_model else None
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description="Roboflow perception worker")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-fps", type=float, default=15)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--detector-model", default="basketball-player-detection-3-ycjdo/13")
    parser.add_argument("--keypoint-model", default="basketball-court-detection-2/14")
    parser.add_argument("--number-model", default="basketball-jersey-numbers-ocr/3")
    parser.add_argument("--number-every", type=int, default=5)
    parser.add_argument("--keypoint-every", type=int, default=3)
    parser.add_argument("--crops", default="", help="JSON со списком рамок номеров для чтения")
    parser.add_argument("--crops-out", default="", help="куда записать прочитанные номера")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--keypoint-confidence", type=float, default=0.3)
    parser.add_argument("--tracker", choices=["none", "sam2"], default="none",
                        help="none — треки ведёт ByteTrack в приложении; sam2 — трекинг выполняется здесь")
    parser.add_argument("--sam2-checkpoint", default="")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam2-reprompt-every", type=float, default=8.0,
                        help="секунд между переподачами промптов в потоковом режиме; "
                             "частая переподача плодит дубликаты треков")
    options = parser.parse_args(argv)

    # frames.py лежит рядом; добавляем корень проекта, не импортируя остальной пакет
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from basketball_cv.frames import plan_frames, iter_frames

    if options.crops:
        read_crops(options)
        return

    info, plan = plan_frames(options.video, options.target_fps, options.start, options.max_seconds, options.max_width)
    models = load_models(options)
    tracker = build_tracker(options)
    out_path = Path(options.out)
    partial = out_path.with_suffix(".partial")
    started = time.monotonic()
    records = []
    frames_dir = None
    player_boxes = {}
    last_prompt, expected_players = -1e9, 0

    if tracker is not None and not tracker.streaming:
        # официальному SAM2 нужна папка с кадрами: готовим её по ходу детекции
        frames_dir = Path(tempfile.mkdtemp(prefix="bcv-frames-"))

    for position, (index, t, frame) in enumerate(iter_frames(options.video, plan)):
        record = process_frame(frame, models, options, position)
        record.update(type="frame", frame=index, time=round(t, 4))
        if tracker is not None and tracker.streaming:
            tracks = tracker.track(frame) if tracker.prompted else []
            live = list(tracks)
            matched, players = attach_tracks(record["detections"], tracks)
            expected_players = max(expected_players, players)
            lost = players and matched < max(1, int(0.7 * min(players, expected_players)))
            if not tracker.prompted or lost or t - last_prompt >= options.sam2_reprompt_every:
                fresh = [d for d in record["detections"]
                         if is_player(d["label"]) and not d.get("from_tracker")]
                if fresh:
                    for detection, track_id in zip(fresh, tracker.prompt(frame, [d["box"] for d in fresh], live)):
                        detection["track"] = track_id
                    last_prompt = t
        elif tracker is not None:
            import cv2

            cv2.imwrite(str(frames_dir / f"{position:06d}.jpg"), frame)
            player_boxes[position] = [d["box"] for d in record["detections"] if is_player(d["label"])]
        records.append(record)
        speed = len(records) / max(0.01, time.monotonic() - started)
        print(f"PROGRESS {len(records)} {plan.expected_frames} {speed:.2f}", file=sys.stderr, flush=True)

    if tracker is not None and not tracker.streaming and records:
        print("SAM2: отслеживание по всему фрагменту", file=sys.stderr, flush=True)
        try:
            tracks = tracker.run(frames_dir, player_boxes, len(records))
            for position, record in enumerate(records):
                attach_tracks(record["detections"], tracks.get(position, []))
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)

    with partial.open("w", encoding="utf-8") as sink:
        header = dict(type="header", video=str(options.video), fps=info["fps"], stride=plan.stride,
                      detector_model=options.detector_model, keypoint_model=options.keypoint_model,
                      number_model=options.number_model, tracker=options.tracker)
        sink.write(json.dumps(header) + "\n")
        for record in records:
            sink.write(json.dumps(record) + "\n")
        sink.write(json.dumps(dict(type="end", frames=len(records))) + "\n")
    count = len(records)
    partial.replace(out_path)  # файл появляется только целиком — кэш не бывает «битым»
    print(f"DONE {count}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
