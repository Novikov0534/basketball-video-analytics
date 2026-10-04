"""Сквозная обработка: видео → восприятие → аналитика → отчёт.

Этапы:
  1. Восприятие. Для backend=roboflow модели Roboflow работают в отдельном
     процессе и сохраняют детекции, ключевые точки площадки и номера в
     кэш JSONL (perception.py). Для yolo/roboflow_api детектор вызывается
     прямо в основном цикле.
  2. Команды. По равномерной выборке кадров обучается кластеризация
     (teams.py) — до основного прохода, чтобы команда была известна сразу.
  3. Основной проход по кадрам: трекинг, гомография, владение, скорость,
     номера, события, отрисовка видео.
  4. Постобработка: сопоставление кластеров с названиями команд по составам,
     объединение треков в игроков, box score, карта бросков, отчёт.
"""

from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import shutil
import subprocess
import time
import uuid

import cv2
import numpy as np

from .analytics import BallSelector, Possession, Statistics
from .boxscore import PLAYER_FIELDS, TEAM_FIELDS, TRACK_FIELDS, build_boxscore
from .detectors import (PlayerTracker, ProvidedTracks, RoboflowApiDetector, YoloCourtKeypoints,
                         YoloDetector)
from .drawing import TEAM_COLORS, UNKNOWN_COLOR, REFEREE_COLOR, ascii_label, heatmap_image, minimap, shot_chart_image
from .events import EventEngine
from .frames import iter_frames, plan_frames, sample_frames, video_info  # noqa: F401 (video_info — публичный API)
from .geometry import KeypointCourtMapper, ManualCourtMapper, NoCourtMapper, SceneCutDetector
from .identity import (JerseyNumberAssigner, match_clusters_to_teams, pair_numbers, select_number_crops)
from .jersey import JerseyNumberClassifier, default_weights
from .ocr import ScoreMonitor, digits, tesseract_command
from .perception import CachedPerception, read_number_crops, run_roboflow_worker
from .report import EVENT_FIELDS, archive_result, make_report, write_csv, write_json
from .teams import TeamClusterer, TrackTeamVoter, collect_crops, make_embedder

MIN_TRACK_SECONDS = 0.5
TRACK_POINT_FIELDS = ["time_s", "frame", "segment", "track_id", "team", "x_px", "y_px", "x_m", "y_m",
                      "speed_kmh", "has_ball", "confidence"]


def first_frame(path, start=0, max_width=1280):
    """Первый кадр выбранного фрагмента для предпросмотра."""
    info = video_info(path)
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(float(start) * info["fps"]))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("Выбранное начало находится за пределами видео.")
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, round(h * max_width / w)))
    return frame, info


class Progress:
    """Делит общий прогресс на этапы: stage(0.0..0.6) → общий масштаб."""

    def __init__(self, callback):
        self.callback = callback
        self.low, self.high = 0.0, 1.0

    def stage(self, low, high):
        self.low, self.high = low, high
        return self

    def __call__(self, fraction, message, frame=None):
        if self.callback:
            self.callback(self.low + (self.high - self.low) * min(1.0, max(0.0, fraction)), message, frame)


# ------------------------------------------------------------------ этап 1
def build_perception(video, cfg, cache_dir, api_key, progress, cancel):
    """→ (детектор, источник ключевых точек или None)."""
    if cfg.backend == "roboflow":
        path = run_roboflow_worker(video, cfg, cache_dir, api_key, progress, cancel)
        perception = CachedPerception(path)
        return perception, (perception.keypoints if perception.provides_keypoints else None)
    if cfg.backend == "roboflow_api":
        return RoboflowApiDetector(cfg, api_key), None
    if cfg.backend == "yolo":
        detector = YoloDetector(cfg)
        keypoints = None
        if cfg.court_keypoint_weights:
            keypoints = YoloCourtKeypoints(cfg.court_keypoint_weights, cfg.device, cfg.image_size).keypoints
        return detector, keypoints
    raise ValueError("Для демонстрации требуется явно переданный детектор (CachedPerception).")


def make_tracker(cfg, detector, fps, id_offset=0):
    """ByteTrack или готовые треки SAM2 из восприятия.

    Если SAM2 запросили, но кэш треков не содержит, продолжаем на ByteTrack:
    система не должна останавливаться из-за необязательной зависимости.
    """
    if cfg.tracker == "sam2" and getattr(detector, "provides_tracks", False):
        return ProvidedTracks(fps, id_offset)
    return PlayerTracker(fps, id_offset)


def make_mapper(cfg, court, frame, keypoint_source):
    if cfg.calibration == "auto" and keypoint_source is not None:
        return KeypointCourtMapper(court), "автоматическая (ключевые точки площадки на каждом кадре)"
    if cfg.image_points:
        return ManualCourtMapper(cfg, court, frame), "ручная (4 точки) + компенсация движения камеры"
    return NoCourtMapper(), "нет"


# ------------------------------------------------------------------ этап 2
def fit_teams(video, plan, cfg, court, detector, keypoint_source, progress, samples=40):
    embedder, warning = make_embedder(cfg.team_embedder, cfg.device)
    clusterer = TeamClusterer(embedder, reducer=cfg.team_reducer)
    crops = []
    for position, (index, t, frame) in enumerate(sample_frames(video, plan, samples)):
        players = [d for d in detector.detect(frame, index) if d.kind == "player"]
        if keypoint_source is not None:
            # отсекаем людей вне площадки (скамейка, зрители) по гомографии этого кадра
            mapper = KeypointCourtMapper(court)
            if mapper.update(t, keypoint_source(frame, index)):
                players = [p for p in players if mapper.project(p.foot) is not None]
        crops.extend(collect_crops(frame, players))
        progress((position + 1) / samples, f"Команды: собрано {len(crops)} изображений игроков")
    crops = crops[:800]
    if not clusterer.fit(crops):
        return clusterer, (warning or "") + " Команды разделить не удалось: мало игроков в выборке.", embedder.name
    return clusterer, warning, embedder.name


# ------------------------------------------------------------------ отрисовка
def draw_frame(frame, state, labels, teams, t, score, team_names, court, banner):
    """Кадр с разметкой: рамки команд, подписи игроков, мяч, счёт, мини-карта."""
    out = frame.copy()
    owner = state["owner"]
    for box in state.get("referees", []):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), REFEREE_COLOR, 2)
        label = "REFEREE"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), REFEREE_COLOR, -1)
        cv2.putText(out, label, (x1 + 3, max(th, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)
    for track, box, speed in state["players"]:
        team = teams.get(track)
        color = TEAM_COLORS[team] if team in (0, 1) else UNKNOWN_COLOR
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if track == owner else 2)
        label = labels.get(track, f"id{track}")
        if speed is not None:
            label += f" {speed * 3.6:.0f}km/h"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, max(th, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if state["ball"] is not None:
        cv2.circle(out, tuple(map(int, state["ball"])), 9, (0, 220, 255), 2, cv2.LINE_AA)
    height, width = out.shape[:2]
    draw_scoreboard(out, team_names, score, t, state["metric"])
    if banner:
        (tw, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (width // 2 - tw // 2 - 12, 52), (width // 2 + tw // 2 + 12, 86), (20, 20, 20), -1)
        cv2.putText(out, banner, (width // 2 - tw // 2, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 220, 255), 2, cv2.LINE_AA)
    court_points = [(xy, teams.get(track), track == owner)
                    for track, xy in state["positions"].items() if xy is not None]
    if court_points:
        small = minimap(court, court_points, width_px=min(300, width // 4))
        h, w = small.shape[:2]
        region = out[height - h - 10: height - 10, width - w - 10: width - 10]
        out[height - h - 10: height - 10, width - w - 10: width - 10] = cv2.addWeighted(region, 0.25, small, 0.75, 0)
    return out


def draw_scoreboard(out, team_names, score, t, metric_valid):
    """Компактное табло: названия команд, счёт и время видео."""
    width = out.shape[1]
    cv2.rectangle(out, (0, 0), (width, 44), (24, 28, 36), -1)
    left, right = ascii_label(team_names[0], 16), ascii_label(team_names[1], 16)
    board = f"{left} {score[0]} : {score[1]} {right}"
    (tw, _), _ = cv2.getTextSize(board, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.putText(out, board, (max(12, width // 2 - tw // 2), 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 220, 150), 2, cv2.LINE_AA)
    cv2.putText(out, f"{t:6.1f} s", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 205, 215), 1, cv2.LINE_AA)


BANNER_TEXT = {
    "shot": "SHOT", "made": "MADE", "steal": "STEAL", "turnover": "TURNOVER",
    "pass": "PASS", "rebound_off": "OFF. REBOUND", "rebound_def": "DEF. REBOUND",
    "block": "BLOCK", "assist": "ASSIST",
}


def event_banner(event, labels):
    """Короткая подпись события для видео."""
    if event.kind == "shot" and event.outcome == "missed":
        title = "MISS"
    elif event.kind == "made":
        title = f"{event.points or 2}PT MADE"
    else:
        title = BANNER_TEXT.get(event.kind, event.kind.upper())
    first = labels.get(event.player_id, f"id{event.player_id}") if event.player_id is not None else ""
    second = labels.get(event.other_player_id, f"id{event.other_player_id}") if event.other_player_id is not None else ""
    if event.kind in ("pass", "assist") and first and second:
        return f"{title} | {first} -> {second}"
    return f"{title} | {first}" if first else title


# ------------------------------------------------------------------ основной вход
def analyze(video, cfg, output_root="outputs", detector=None, keypoint_source=None, api_key=None,
            progress=None, cancel=None):
    """Два прохода по кадрам.

    Первый — анализ: трекинг, гомография, владение, события. Видео при этом не
    рисуется, потому что личности игроков (номер, имя, команда) становятся
    известны только к концу. Затем читаются номера и собирается box score.
    Второй проход — отрисовка: подписи и счёт верны с первого кадра.
    """
    cfg.validate()
    court = cfg.court
    info, plan = plan_frames(video, cfg.target_fps, cfg.start_seconds, cfg.max_seconds, cfg.max_width)
    fps = plan.analysis_fps
    output_root = Path(output_root).resolve()
    stages = Progress(progress)
    heavy = cfg.backend == "roboflow" and detector is None

    started = time.monotonic()
    if detector is None:
        detector, keypoint_source = build_perception(video, cfg, output_root / "_cache", api_key,
                                                     stages.stage(0.0, 0.45), cancel)
    elif keypoint_source is None and getattr(detector, "provides_keypoints", False):
        keypoint_source = detector.keypoints
    has_actions = bool(getattr(detector, "has_actions", False))

    folder = output_root / ("run-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    cfg.save(folder / "config.json")

    base = 0.45 if heavy else 0.0
    clusterer, team_warning, embedder_name = fit_teams(
        video, plan, cfg, court, detector, keypoint_source, stages.stage(base, base + 0.05))
    analysis_progress = stages.stage(base + 0.05, 0.75 if heavy else 0.8)

    tracker = make_tracker(cfg, detector, fps)
    tracker_source = tracker.source
    voter = TrackTeamVoter()
    last_team_vote = {}
    numbers = JerseyNumberAssigner(min_votes=2)
    possession, selector = Possession(), BallSelector()
    stats = Statistics(cfg, 1 / fps)
    events = EventEngine(court, fps, has_actions=has_actions)
    scoreboard = ScoreMonitor(cfg.initial_score)
    cuts = SceneCutDetector()
    mapper, calibration_mode = None, "нет"
    heat_points, track_points, frame_states, number_requests = [], [], [], []
    segment, processed, ball_frames, metric_frames, referee_frames, max_referees = 0, 0, 0, 0, 0, 0
    last_ocr, last_progress, last_time = -100.0, 0.0, cfg.start_seconds
    permutation, stopped, error = [0, 1], False, None
    previous_boxes = []
    # номера, прочитанные прямо в детекциях (демо и старые кэши), читать заново не нужно
    inline_numbers = cfg.backend != "roboflow"

    try:
        for index, t, frame in iter_frames(video, plan, cancel):
            height, width = frame.shape[:2]
            last_time = t
            cut = cuts.update(frame) if processed else False
            if mapper is None:
                mapper, calibration_mode = make_mapper(cfg, court, frame, keypoint_source)
                cv2.imwrite(str(folder / "first_frame.jpg"), frame)
            if cut:
                segment += 1
                tracker = make_tracker(cfg, detector, fps, id_offset=max(stats.players, default=0) + 100)
                possession, selector = Possession(), BallSelector()
                events.reset(t)

            detections = detector.detect(frame, index)
            keypoints = keypoint_source(frame, index) if keypoint_source is not None else None
            mapper.update(t, keypoints=keypoints, cut=cut, frame=frame, boxes=previous_boxes)

            candidates = [d for d in detections if d.kind == "player"]
            referees = [d for d in detections if d.kind == "referee"]
            if mapper.valid:
                # Людей у скамейки/технической зоны детектор иногда принимает за судей.
                # При надёжной гомографии оставляем только тех, чья опорная точка
                # проецируется на игровую площадку.
                candidates = [d for d in candidates if mapper.project(d.foot) is not None]
                referees = [d for d in referees if mapper.project(d.foot) is not None]
            referee_frames += int(bool(referees))
            max_referees = max(max_referees, len(referees))
            players = tracker.update(candidates)
            previous_boxes = [p.box for p in players]

            # команды: новый голос для трека не чаще раза в 0.4 с — экономим SigLIP
            due = [p for p in players if t - last_team_vote.get(p.track_id, -1.0) >= 0.4]
            if due and clusterer.fitted:
                crops, owners = [], []
                for p in due:
                    crop = collect_crops(frame, [p])
                    if crop:
                        crops.append(crop[0])
                        owners.append(p.track_id)
                for track, (label, margin) in zip(owners, clusterer.predict(crops)):
                    voter.add(track, label, margin)
                    last_team_vote[track] = t
            # clusters — «сырые» номера кластеров: в них ведутся события и статистика,
            # названия команд применяются в конце, когда известно соответствие составам
            clusters = {p.track_id: voter.team(p.track_id) for p in players}

            number_boxes = [d for d in detections if d.kind == "number"]
            if cfg.read_numbers and number_boxes:
                if inline_numbers:
                    numbers.update(players, number_boxes)
                else:
                    for track, detection in pair_numbers(players, number_boxes):
                        x1, y1, x2, y2 = [int(round(v)) for v in detection.box]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size:
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                            contrast = float(gray.std())
                        else:
                            sharpness = contrast = 0.0
                        number_requests.append(dict(
                            track_id=track, frame=index, time_s=round(t, 3),
                            box=[round(v, 1) for v in detection.box], area=round(detection.area, 1),
                            sharpness=round(sharpness, 2), contrast=round(contrast, 2),
                            detector_confidence=round(float(detection.confidence), 3),
                        ))
            positions = {p.track_id: mapper.project(p.foot) for p in players}
            ball = selector.choose(detections, t, frame.shape)
            owner = possession.update(t, players, ball)
            speeds = stats.update(t, players, clusters, positions, owner, possession.observed)
            hoops = [d.box for d in detections if d.kind == "hoop"]
            hoops += [tuple(np.array(r) * [width, height, width, height]) for r in cfg.hoops]
            events.update(t, players, clusters, owner, possession.observed, ball, detections, hoops, positions)

            if cfg.score_rois and "A" in cfg.score_rois and "B" in cfg.score_rois \
                    and t - last_ocr >= cfg.ocr_interval and tesseract_command():
                last_ocr = t
                values = []
                for key in ("A", "B"):
                    x1, y1, x2, y2 = np.array(cfg.score_rois[key]) * [width, height, width, height]
                    values.append(digits(frame[int(y1):int(y2), int(x1):int(x2)]))
                for team, delta in scoreboard.observe(t, values):
                    events.add(t, "score_change", team=team, points=delta, confidence=0.9, status="ocr_observed",
                               reason="Изменение табло подтверждено несколькими кадрами (сверка).")

            for p in players:
                xy = positions.get(p.track_id)
                if xy is not None:
                    heat_points.append((xy[0], xy[1], 1.0 / fps))
                track_points.append(dict(
                    time_s=round(t, 4), frame=index, segment=segment, track_id=p.track_id,
                    cluster=voter.team(p.track_id), x_px=round(p.foot[0], 1), y_px=round(p.foot[1], 1),
                    x_m=round(xy[0], 3) if xy else None, y_m=round(xy[1], 3) if xy else None,
                    speed_kmh=round(speeds[p.track_id] * 3.6, 2) if speeds.get(p.track_id) is not None else None,
                    has_ball=int(owner == p.track_id and possession.observed), confidence=round(p.confidence, 3),
                ))
            frame_states.append(dict(
                frame=index, t=t, owner=owner, metric=mapper.valid,
                ball=tuple(map(float, ball.center)) if ball is not None else None,
                referees=[tuple(map(float, r.box)) for r in referees],
                players=[(p.track_id, tuple(map(float, p.box)), speeds.get(p.track_id)) for p in players],
                positions={p.track_id: positions.get(p.track_id) for p in players},
            ))

            processed += 1
            ball_frames += int(ball is not None)
            metric_frames += int(mapper.valid)
            if time.monotonic() - last_progress > 0.5:
                done = (index - plan.start_frame) / plan.stride + 1
                analysis_progress(done / plan.expected_frames,
                                  f"Анализ: {processed} кадров · {processed / max(0.01, time.monotonic() - started):.1f} кадр/с")
                last_progress = time.monotonic()
        stopped = bool(cancel is not None and cancel.is_set())
    except Exception as exc:  # причина сохраняется в папке результата
        error = exc
    if error is not None:
        write_json(folder / "failure.json", dict(error=str(error), processed_frames=processed))
        raise error
    if processed == 0:
        raise ValueError("Нет кадров для обработки.")

    # ------------------------------------------- номера
    number_reader, number_warning = "—", None
    if cfg.read_numbers and number_requests and not stopped:
        number_reader, number_warning = read_numbers_for_tracks(
            video, plan, cfg, numbers, number_requests, api_key,
            stages.stage(0.75, 0.85), cancel, output_root / "_cache", folder / "jersey_debug")

    # ------------------------------------------- личности, команды, статистика
    permutation = cluster_permutation(voter, numbers, stats, cfg)
    matched = permutation_from_rosters(voter, numbers, cfg)
    if cfg.swap_teams and not matched:
        permutation = [1, 0]
    mapping_source = "составы команд (совпадение номеров)" if matched else (
        "ручная перестановка" if cfg.swap_teams else "порядок кластеров (без привязки к составам)")
    remap_events(events.events, permutation, recorded=[0, 1])

    confirmed_numbers = numbers.confirmed()
    track_rows = []
    for row in stats.rows():
        # relaxed: короткие треки тоже получают команду, иначе их события
        # остаются без принадлежности и очки не попадают в счёт
        cluster = voter.team(row["track_id"], relaxed=True)
        row["team"] = None if cluster is None else permutation[cluster]
        row["number"] = confirmed_numbers.get(row["track_id"], "")
        track_rows.append(row)
    # обрывки треков короче полсекунды — шум ByteTrack, в таблицу игроков не идут
    named_rows = [r for r in track_rows if (r["visible_s"] or 0) >= MIN_TRACK_SECONDS or r["number"]]
    fill_missing_event_teams(events.events, {r["track_id"]: r["team"] for r in track_rows})
    team_score = [0, 0]
    for event in events.events:
        if event.kind == "made" and event.team in (0, 1):
            team_score[event.team] += event.points or 2
    events.unassigned_points = sum(e.points or 2 for e in events.events
                                   if e.kind == "made" and e.team not in (0, 1))
    players_table, teams_table, track_to_player = build_boxscore(
        named_rows, events.events, cfg.team_names, cfg.rosters)
    for row in track_rows:
        row["player"] = track_to_player.get(row["track_id"], "")
    for point in track_points:
        cluster = point.pop("cluster")
        point["team"] = None if cluster is None else permutation[cluster]

    # ------------------------------------------- второй проход: отрисовка
    labels = frame_labels(track_rows, players_table, track_to_player)
    teams_by_track = {r["track_id"]: r["team"] for r in track_rows}
    # отрисовку не прерываем: она дешёвая, а частичный результат должен быть с видео
    output_fps = render_video(video, plan, cfg, folder, frame_states, labels, teams_by_track, events.events,
                              cfg.team_names, court, stages.stage(0.85, 0.98))
    encode_warning = encode_video(folder)

    warnings = collect_warnings(cfg, has_actions, team_warning or number_warning, encode_warning, ball_frames, metric_frames,
                                processed, mapper, events, confirmed_numbers, mapping_source, tracker_source)
    summary = dict(
        source_name=Path(video).name, source_info=info, backend=cfg.backend,
        models=dict(detector=cfg.detector_model_id if cfg.backend.startswith("roboflow") else cfg.weights,
                    court_keypoints=cfg.keypoint_model_id if cfg.backend == "roboflow" else cfg.court_keypoint_weights,
                    numbers=cfg.number_model_id if cfg.backend == "roboflow" and cfg.read_numbers else ""),
        league=cfg.league, calibration=calibration_mode, calibration_quality=mapper.quality(),
        team_embedder=embedder_name, team_reducer=cfg.team_reducer, number_reader=number_reader,
        tracker=tracker_source, tracker_requested=cfg.tracker,
        team_mapping=mapping_source, team_names=cfg.team_names,
        processed_frames=processed, processed_seconds=round(processed / fps, 3), analysis_fps=round(fps, 3),
        output_fps=round(output_fps, 3),
        start_seconds=cfg.start_seconds, last_timestamp_s=round(last_time, 3),
        wall_seconds=round(time.monotonic() - started, 2), cancelled=stopped,
        score=team_score, score_source="попадания (2 или 3 очка по зоне броска)",
        score_timeline=[dict(time_s=e.time_s, team=e.team, points=e.points) for e in events.events
                        if e.kind == "made"],
        unassigned_points=events.unassigned_points, scoreboard_ocr=scoreboard.score,
        ball_detection_fraction=round(ball_frames / processed, 3),
        referee_detection_fraction=round(referee_frames / processed, 3),
        max_referees_seen=max_referees,
        calibrated_frame_fraction=round(metric_frames / processed, 3),
        number_crops_read=(len(select_number_crops(number_requests, cfg.number_crops_per_track))
                           if number_requests else 0),
        number_crops_seen=len(number_requests),
        track_count=len(track_rows), identified_tracks=len(confirmed_numbers),
        player_count=len(players_table), event_count=len(events.events),
        warnings=warnings, versions=package_versions(), reviewed=False,
    )
    event_rows = [asdict(e) for e in events.events]
    write_csv(folder / "players.csv", players_table, PLAYER_FIELDS)
    write_csv(folder / "teams.csv", teams_table, TEAM_FIELDS)
    write_csv(folder / "tracks_summary.csv", track_rows, TRACK_FIELDS)
    write_csv(folder / "tracks.csv", track_points, TRACK_POINT_FIELDS)
    write_csv(folder / "events.csv", event_rows, EVENT_FIELDS)
    write_csv(folder / "score_ocr.csv", scoreboard.rows, ["time_s", "score_a", "score_b", "source"])
    write_json(folder / "events.json", event_rows)
    write_json(folder / "summary.json", summary)
    cv2.imwrite(str(folder / "heatmap.png"), heatmap_image(court, heat_points))
    cv2.imwrite(str(folder / "shot_chart.png"), shot_chart_image(court, shot_list(event_rows)))
    make_report(folder, summary, players_table, teams_table, event_rows)
    archive_result(folder)
    return folder, summary


def read_numbers_for_tracks(video, plan, cfg, numbers, requests, api_key, progress, cancel, cache_dir, debug_dir=None):
    """Читает номера доступной моделью. → (какая модель, предупреждение).

    Обоим reader-ам отдаётся только выборка лучших кропов на игрока. Это
    исключает сотни повторов одного и того же номера и ускоряет обработку,
    сохраняя несколько разных ракурсов для устойчивого голосования.
    """
    classifier, warning = None, None
    if cfg.number_reader in ("auto", "resnet"):
        weights = Path(cfg.number_weights) if cfg.number_weights else default_weights()
        if weights and Path(weights).is_file():
            try:
                classifier = JerseyNumberClassifier(weights, cfg.device)
            except Exception as exc:
                if cfg.number_reader == "resnet":
                    raise
                classifier = None
                warning = f"ResNet для номеров не загрузился ({type(exc).__name__}); читает SmolVLM2."
        elif cfg.number_reader == "resnet":
            raise ValueError("Не найдены веса ResNet: обучите их scripts/train_jersey_resnet.py "
                             "или укажите number_weights.")
    selected = select_number_crops(requests, cfg.number_crops_per_track)
    if classifier is not None:
        read_all_number_crops(video, plan, selected, classifier, numbers, progress, cancel, debug_dir)
        return classifier.name, warning
    for track, text in read_number_crops(video, cfg, selected,
                                         api_key, progress, cancel, cache_dir):
        numbers.vote(track, text)
    return "smolvlm", warning


def read_all_number_crops(video, plan, requests, classifier, numbers, progress=None, cancel=None, debug_dir=None):
    """Читает заранее выбранные лучшие кропы номера пачками ResNet."""
    wanted = {}
    for request in requests:
        wanted.setdefault(int(request["frame"]), []).append(request)
    debug_rows = []
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
    done, total = 0, max(1, len(requests))
    for index, _, frame in iter_frames(video, plan, cancel):
        items = wanted.get(index)
        if not items:
            continue
        height, width = frame.shape[:2]
        crops, owners, crop_items = [], [], []
        for item in items:
            x1, y1, x2, y2 = item["box"]
            x1, y1 = max(0, int(x1) - 4), max(0, int(y1) - 4)
            x2, y2 = min(width, int(x2) + 4), min(height, int(y2) + 4)
            if x2 > x1 and y2 > y1:
                crops.append(frame[y1:y2, x1:x2])
                owners.append(item["track_id"])
                crop_items.append(item)
        predictions = classifier.predict(crops)
        for crop, track, item, (text, confidence) in zip(crops, owners, crop_items, predictions):
            if text:
                numbers.vote(track, text, confidence)
            if debug_dir is not None:
                label = text or "unknown"
                filename = f"track_{track}_frame_{index}_{label}_{confidence:.3f}.jpg"
                cv2.imwrite(str(debug_dir / filename), crop)
                debug_rows.append(dict(
                    track_id=track, frame=index, time_s=item.get("time_s", ""),
                    prediction=label, confidence=confidence,
                    quality_score=item.get("quality_score", ""),
                    sharpness=item.get("sharpness", ""), contrast=item.get("contrast", ""),
                    detector_confidence=item.get("detector_confidence", ""), filename=filename,
                ))
        done += len(items)
        if progress:
            progress(done / total, f"Номера (ResNet): {done}/{total} рамок")
    if debug_dir is not None and debug_rows:
        write_csv(debug_dir / "predictions.csv", debug_rows, [
            "track_id", "frame", "time_s", "prediction", "confidence", "quality_score",
            "sharpness", "contrast", "detector_confidence", "filename",
        ])


def fill_missing_event_teams(events, team_by_track):
    """Проставляет команду событиям, у которых она не определилась во время прохода.

    Команда трека уточняется до конца обработки: короткий трек может получить
    её позже, чем произошло событие. Здесь событие и трек сводятся обратно,
    иначе очки не попадают в счёт команды.
    """
    for event in events:
        if event.team is None and event.player_id is not None:
            event.team = team_by_track.get(event.player_id)
    return events


def frame_labels(track_rows, players_table, track_to_player):
    """Подпись трека на видео: номер и имя, если игрок опознан."""
    names = {row["player"]: row for row in players_table}
    labels = {}
    for row in track_rows:
        identity = names.get(track_to_player.get(row["track_id"], ""))
        if identity and identity["number"]:
            name = ascii_label(identity["name"], 14)
            labels[row["track_id"]] = f"#{identity['number']}" + (f" {name}" if name else "")
        else:
            labels[row["track_id"]] = f"id{row['track_id']}"
    return labels


def render_video(video, analysis_plan, cfg, folder, frame_states, labels, teams, events, team_names, court,
                 progress=None):
    """Второй проход: плавное видео до 30 FPS поверх разреженного анализа.

    Тяжёлые модели работают с ``cfg.target_fps`` (по умолчанию 30 FPS), а
    итоговый ролик декодируется с частотой до ``cfg.output_fps``. Для каждого
    выходного кадра берётся ближайшее состояние анализа. Если анализ выполняется
    реже исходного видео, соседние выходные кадры используют ближайшее состояние
    трекера.
    """
    made = sorted((e.time_s, e.team, e.points or 0) for e in events if e.kind == "made")
    banners = [(e.time_s, event_banner(e, labels)) for e in events if e.kind in BANNER_TEXT]
    _, output_plan = plan_frames(video, cfg.output_fps, cfg.start_seconds, cfg.max_seconds, cfg.max_width)
    writer, total = None, max(1, output_plan.expected_frames)
    state_index = 0

    try:
        last_state_time = frame_states[-1]["t"]
        tail = 1.0 / max(1e-6, analysis_plan.analysis_fps)
        for position, (_, t, frame) in enumerate(iter_frames(video, output_plan)):
            if t > last_state_time + tail:
                break
            while state_index + 1 < len(frame_states):
                current = frame_states[state_index]
                following = frame_states[state_index + 1]
                if abs(following["t"] - t) <= abs(current["t"] - t):
                    state_index += 1
                else:
                    break
            state = frame_states[state_index]
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(folder / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         output_plan.analysis_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Не удалось создать выходное видео.")
            score = [0, 0]
            for time_s, team, points in made:
                if time_s <= t and team in (0, 1):
                    score[team] += points
            banner = next((text for time_s, text in reversed(banners) if time_s <= t < time_s + 1.2), "")
            writer.write(draw_frame(frame, state, labels, teams, t, score, team_names, court, banner))
            if progress and position % 15 == 0:
                progress(position / total, f"Отрисовка видео: {position}/{total}")
    finally:
        if writer is not None:
            writer.release()
    return output_plan.analysis_fps


# ------------------------------------------------------------------ помощники

def remap_events(events, permutation, recorded):
    """В событиях команда записана как кластер; переводим в индекс команды."""
    if permutation == recorded:
        return
    for event in events:
        if event.team in (0, 1) and event.kind != "score_change":  # табло уже в порядке A/B
            event.team = permutation[event.team]


def _cluster_numbers(voter, numbers):
    result = {0: set(), 1: set()}
    for track, number in numbers.confirmed().items():
        cluster = voter.team(track)
        if cluster in (0, 1):
            result[cluster].add(number)
    return result


def permutation_from_rosters(voter, numbers, cfg):
    permutation, _ = match_clusters_to_teams(_cluster_numbers(voter, numbers), cfg.rosters)
    return permutation


def cluster_permutation(voter, numbers, stats, cfg):
    permutation = permutation_from_rosters(voter, numbers, cfg)
    if permutation is not None:
        return permutation
    return [1, 0] if cfg.swap_teams else [0, 1]


def shot_list(event_rows):
    shots = []
    for e in event_rows:
        if e["kind"] == "shot" and e.get("points") != 1:
            shots.append(dict(x_m=e["x_m"], y_m=e["y_m"], team=e["team"], made=e["outcome"] == "made"))
    return shots


def encode_video(folder):
    raw, target = folder / "raw.mp4", folder / "annotated.mp4"
    if not raw.exists():
        return "Видео не создано: обработка остановлена до отрисовки."
    ffmpeg = shutil.which("ffmpeg")
    warning = None
    if ffmpeg:
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-an", "-c:v", "libx264", "-preset", "fast",
                   "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
        if subprocess.run(command, capture_output=True, text=True).returncode:
            warning = "FFmpeg не перекодировал видео в H.264; сохранён исходный mp4v."
            shutil.copy2(raw, target)
    else:
        shutil.copy2(raw, target)
        warning = "FFmpeg отсутствует: браузер может не воспроизвести MP4; откройте файл в VLC."
    raw.unlink(missing_ok=True)
    return warning


def collect_warnings(cfg, has_actions, team_warning, encode_warning, ball_frames, metric_frames, processed,
                     mapper, events, confirmed_numbers, mapping_source, tracker_source="bytetrack"):
    warnings = [w for w in (team_warning, encode_warning) if w]
    if cfg.tracker == "sam2" and tracker_source != "sam2":
        warnings.append("SAM2 запрошен, но недоступен: отслеживание выполнил ByteTrack. "
                        "Установка: scripts/bootstrap.py --roboflow --sam2.")
    if cfg.backend == "yolo" and cfg.weights.startswith("yolo11"):
        warnings.append("COCO-веса YOLO — базовая линия: нет классов мяча-в-корзине, бросков, судей и кольца. "
                        "Для баскетбола используйте Roboflow или свои веса.")
    if not has_actions:
        warnings.append("Детектор без классов действий: броски определяются по полёту мяча к кольцу (менее надёжно).")
    if ball_frames / processed < 0.3:
        warnings.append(f"Мяч найден лишь в {100 * ball_frames / processed:.0f}% кадров: владение и события неполны.")
    if metric_frames / processed < 0.5:
        warnings.append(f"Метры доступны только в {100 * metric_frames / processed:.0f}% кадров: "
                        "площадка не распознана (крупные планы, повторы) или калибровка не задана.")
    warnings.extend(sorted(getattr(mapper, "reasons", set())))
    if events.unassigned_points:
        warnings.append(f"{events.unassigned_points} очк. не отнесены к команде: команда бросавшего не определена.")
    if cfg.read_numbers and not confirmed_numbers:
        warnings.append("Номера игроков не подтверждены: статистика ведётся по трекам.")
    if "без привязки" in mapping_source:
        warnings.append("Названия команд назначены по порядку кластеров. Для привязки к конкретным командам задайте составы до запуска анализа.")
    warnings.append("Дистанция и владение рассчитываются только по наблюдаемым интервалам. "
                    "События сформированы автоматически; качество оценивается отдельно по ground truth.")
    return warnings


def package_versions():
    versions = {}
    for package in ("torch", "ultralytics", "supervision", "transformers", "opencv-python", "numpy", "gradio"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions

