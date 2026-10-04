"""Синтетический ролик для проверки ПРОГРАММЫ (не точности нейросетей).

Рисуется площадка NBA сверху, по ней по сценарию двигаются 4 игрока, судья и
мяч. Вместо нейросетей записывается файл восприятия в том же формате, что
выдаёт perception_worker (Roboflow): детекции с исходными именами классов,
33 ключевые точки площадки и прочитанные номера. Поэтому демо проходит
ровно тот же автоматический путь, что и реальное видео.

Сценарий (известная «истина» пишется в demo_truth.json):
  0.0–2.5  синий №7 владеет мячом          2.5–2.8  передача №7 → №0
  2.8–5.0  синий №0                         5.0      перехват оранжевого №3
  5.0–6.0  оранжевый №3                     6.0–6.3  передача №3 → №11
  7.0      №11 бросает из-за дуги           7.8      попадание (+3), передача №3 — результативная
  10.0     синий №7 вводит мяч              12.0     №7 бросает из-под кольца (2 очка) — промах
  13.0     подбор в защите: оранжевый №11
"""

from pathlib import Path
import json
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basketball_cv.config import Config  # noqa: E402
from basketball_cv.court import COURTS  # noqa: E402
from basketball_cv.drawing import CourtCanvas  # noqa: E402

WIDTH, HEIGHT, FPS, DURATION = 960, 560, 30, 15.0
OFFSET = np.array([30, 74])
COURT = COURTS["nba"]
CANVAS = CourtCanvas(COURT, width_px=900, margin=0)
BLUE, ORANGE, GREY = (200, 110, 30), (30, 120, 235), (120, 120, 120)

# игрок: (команда, номер, цвет, опорные точки траектории (t, x_m, y_m))
PLAYERS = {
    "A7": (0, "7", BLUE, [(0, 8.0, 5.0), (2.5, 10.0, 6.0), (5, 12.0, 4.0), (10, 12.5, 3.5), (12, 5.0, 7.62), (15, 7.0, 9.0)]),
    "A0": (0, "0", BLUE, [(0, 14.0, 10.0), (2.8, 15.0, 10.5), (5, 16.0, 9.0), (15, 12.0, 8.0)]),
    "B3": (1, "3", ORANGE, [(0, 16.3, 9.2), (5, 16.3, 9.2), (6, 18.0, 11.0), (15, 22.0, 11.0)]),
    "B11": (1, "11", ORANGE, [(0, 20.0, 4.0), (6.3, 19.5, 6.0), (7.0, 19.5, 7.62), (8, 19.0, 7.0), (10, 10.0, 6.0),
                             (13, 4.0, 5.0), (15, 8.0, 6.0)]),
}
REFEREE_PATH = [(0, 10.0, 14.9), (15, 18.0, 14.9)]

# владелец мяча по интервалам (вне интервалов мяч в полёте или без владельца)
OWNERS = [(0, 2.5, "A7"), (2.8, 5.0, "A0"), (5.0, 6.0, "B3"), (6.3, 7.0, "B11"), (10.0, 12.0, "A7"), (13.0, 15.0, "B11")]
SHOTS = [dict(player="B11", start=7.0, action_end=7.3, made=True, basket_from=7.8, basket_to=8.0),
         dict(player="A7", start=12.0, action_end=12.3, made=False)]
TRUTH_EVENTS = [
    dict(time_s=2.8, kind="pass", player="A7"), dict(time_s=5.0, kind="turnover", player="A0"),
    dict(time_s=5.0, kind="steal", player="B3"), dict(time_s=6.3, kind="pass", player="B3"),
    dict(time_s=7.0, kind="shot", player="B11", points=3), dict(time_s=7.8, kind="made", player="B11", points=3),
    dict(time_s=7.8, kind="assist", player="B3"), dict(time_s=12.0, kind="shot", player="A7", points=2),
    dict(time_s=13.0, kind="rebound_def", player="B11"),
]


def path_position(path, t):
    times = [p[0] for p in path]
    return float(np.interp(t, times, [p[1] for p in path])), float(np.interp(t, times, [p[2] for p in path]))


def to_px(xy):
    return np.array(CANVAS.to_px(xy), dtype=float) + OFFSET


def owner_at(t):
    for start, end, player in OWNERS:
        if start <= t < end:
            return player
    return None


def ball_position(t, feet):
    """Мяч: у владельца, в полёте (передача/бросок) или не виден."""
    hand = np.array([14, -40])
    owner = owner_at(t)
    if owner is not None:
        return tuple(feet[owner] + hand)
    for start, end, a, b in ((2.5, 2.8, "A7", "A0"), (6.0, 6.3, "B3", "B11")):
        if start <= t < end:
            alpha = (t - start) / (end - start)
            return tuple(feet[a] + hand + alpha * (feet[b] - feet[a]))
    rim_left, rim_right = to_px(COURT.basket_centers()[0]), to_px(COURT.basket_centers()[1])
    if 7.0 <= t < 7.8:
        alpha = (t - 7.0) / 0.8
        start = to_px(path_position(PLAYERS["B11"][3], 7.0)) + hand
        point = start + alpha * (rim_right - start)
        return (point[0], point[1] - 60 * np.sin(alpha * np.pi))
    if 7.8 <= t < 8.0:
        return tuple(rim_right + (0, 3))
    if 12.0 <= t < 13.0:
        start = to_px(path_position(PLAYERS["A7"][3], 12.0)) + hand
        if t < 12.7:
            alpha = (t - 12.0) / 0.7
            point = start + alpha * (rim_left - start)
            return (point[0], point[1] - 40 * np.sin(alpha * np.pi))
        alpha = (t - 12.7) / 0.3  # отскок от кольца к №11
        target = feet["B11"] + hand
        return tuple(rim_left + alpha * (target - rim_left))
    return None


def draw_player(frame, foot, color, number):
    x, y = int(foot[0]), int(foot[1])
    cv2.rectangle(frame, (x - 14, y - 62), (x + 14, y - 22), color, -1)
    cv2.circle(frame, (x, y - 70), 8, (170, 195, 220), -1)
    cv2.line(frame, (x - 8, y - 22), (x - 8, y), (40, 40, 40), 5)
    cv2.line(frame, (x + 8, y - 22), (x + 8, y), (40, 40, 40), 5)
    if number:
        cv2.putText(frame, number, (x - 7 * len(number), y - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return [x - 16, y - 80, x + 16, y], [x - 12, y - 58, x + 12, y - 26]


def path_length(path, step=0.01):
    points = [path_position(path, t) for t in np.arange(0, DURATION + step, step)]
    return float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])))


def generate(destination=ROOT / "examples"):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    video_path = destination / "demo.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Не удалось создать демо-видео.")
    rng = np.random.default_rng(7)
    background = np.full((HEIGHT, WIDTH, 3), (28, 24, 20), np.uint8)
    floor = np.full((CANVAS.height, CANVAS.width, 3), (95, 140, 176), np.uint8)
    CANVAS.draw_lines(floor)
    background[OFFSET[1]: OFFSET[1] + floor.shape[0], OFFSET[0]: OFFSET[0] + floor.shape[1]] = floor
    keypoints_px = [to_px(v) for v in COURT.vertices()]
    frames = []
    for index in range(int(DURATION * FPS)):
        t = index / FPS
        frame = background.copy()
        cv2.putText(frame, "SYNTHETIC DEMO - software check, not model accuracy", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
        for x, value in ((740, "0"), (850, "3" if t >= 8.3 else "0")):  # табло для сверки OCR
            cv2.rectangle(frame, (x, 4), (x + 80, 62), (255, 255, 255), -1)
            cv2.putText(frame, value, (x + 22, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        feet = {name: to_px(path_position(spec[3], t)) + rng.normal(0, 0.8, 2) for name, spec in PLAYERS.items()}
        owner = owner_at(t)
        detections = []
        for name, (team, number, color, path) in PLAYERS.items():
            box, number_box = draw_player(frame, feet[name], color, number)
            label = "player-in-possession" if name == owner else "player"
            for shot in SHOTS:
                if shot["player"] == name and shot["start"] <= t < shot["action_end"]:
                    label = "player-jump-shot"
            detections.append(dict(label=label, box=box, conf=0.95))
            detections.append(dict(label="number", box=number_box, conf=0.9, text=number))
        box, _ = draw_player(frame, to_px(path_position(REFEREE_PATH, t)), GREY, "")
        detections.append(dict(label="referee", box=box, conf=0.9))
        for rim in COURT.basket_centers():
            x, y = to_px(rim)
            cv2.ellipse(frame, (int(x), int(y)), (10, 5), 0, 0, 360, (40, 70, 240), 2)
            detections.append(dict(label="rim", box=[x - 12, y - 7, x + 12, y + 7], conf=0.9))
        ball = ball_position(t, feet)
        if ball is not None:
            bx, by = map(float, ball)
            cv2.circle(frame, (int(bx), int(by)), 6, (25, 130, 250), -1)
            detections.append(dict(label="ball", box=[bx - 6, by - 6, bx + 6, by + 6], conf=0.9))
        for shot in SHOTS:
            if shot["made"] and shot["basket_from"] <= t < shot["basket_to"]:
                x, y = to_px(COURT.basket_centers()[1])
                detections.append(dict(label="ball-in-basket", box=[x - 14, y - 10, x + 14, y + 12], conf=0.9))
        keypoints = [[round(float(x), 2), round(float(y), 2), 0.2 if i in (12, 18) else 0.92]
                     for i, (x, y) in enumerate(keypoints_px)]  # 2 точки с низкой уверенностью
        for d in detections:
            d["box"] = [round(float(v), 2) for v in d["box"]]
        frames.append(dict(type="frame", frame=index, time=round(t, 4), detections=detections, keypoints=keypoints))
        writer.write(frame)
    writer.release()
    if shutil.which("ffmpeg"):
        temporary = destination / "demo-h264.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)], check=True)
        temporary.replace(video_path)
    with (destination / "demo_perception.jsonl").open("w", encoding="utf-8") as sink:
        sink.write(json.dumps(dict(type="header", synthetic=True, fps=FPS)) + "\n")
        for record in frames:
            sink.write(json.dumps(record, separators=(",", ":")) + "\n")
        sink.write(json.dumps(dict(type="end", frames=len(frames))) + "\n")
    truth = dict(
        score=[0, 3], team_names=["Синие", "Оранжевые"], events=TRUTH_EVENTS,
        distance_m={name: round(path_length(spec[3]), 2) for name, spec in PLAYERS.items()},
        numbers={name: spec[1] for name, spec in PLAYERS.items()},
    )
    (destination / "demo_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = Config(
        backend="replay", league="nba", team_names=["Синие", "Оранжевые"],
        rosters=[{"7": "Иванов", "0": "Петров"}, {"3": "Смирнов", "11": "Сидоров"}],
        team_embedder="color", max_seconds=DURATION, target_fps=15,
        score_rois={"A": [740 / WIDTH, 4 / HEIGHT, 820 / WIDTH, 62 / HEIGHT],
                    "B": [850 / WIDTH, 4 / HEIGHT, 930 / WIDTH, 62 / HEIGHT]},
    )
    cfg.save(destination / "demo_config.json")
    Config().save(destination / "config.example.json")
    return destination


if __name__ == "__main__":
    print(generate())
