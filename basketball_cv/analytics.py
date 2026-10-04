"""Мяч, владение, скорость и дистанция по трекам."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .detectors import Detection


class BallSelector:
    """Выбирает один мяч в кадре: ближайший к предыдущему положению.

    Защита от ложных срабатываний: если кандидат «прыгнул» больше чем на
    35 % диагонали кадра за < 0.5 с, он отбрасывается.
    """

    def __init__(self, hold_seconds=0.3):
        self.hold_seconds = hold_seconds
        self.previous, self.last_time = None, -100.0

    def choose(self, detections, t, shape):
        candidates = [d for d in detections if d.kind == "ball"]
        if not candidates:
            # мяч детектится не в каждом кадре; короткий пропуск закрываем последним
            # известным положением, иначе владение рвётся по несколько раз в секунду
            if self.previous is not None and t - self.last_time <= self.hold_seconds:
                x, y = self.previous
                return Detection((x - 6, y - 6, x + 6, y + 6), 0.3, "ball")
            return None
        diagonal = float(np.hypot(*shape[:2]))
        if self.previous is not None and t - self.last_time < 0.5:
            def score(d):
                return np.linalg.norm(np.array(d.center) - self.previous) / diagonal - 0.15 * d.confidence

            ball = min(candidates, key=score)
            if np.linalg.norm(np.array(ball.center) - self.previous) > 0.35 * diagonal:
                return None
        else:
            ball = max(candidates, key=lambda d: d.confidence)
        self.previous, self.last_time = np.array(ball.center), t
        return ball


class Possession:
    """Кто владеет мячом, с защитой от «мигания».

    Источники (в порядке надёжности):
      1. класс детектора `player-in-possession` (модель Roboflow);
      2. близость мяча к «точке обработки мяча» игрока — середине рамки по
         ширине и 0.55 её высоты от верха (грудь / пояс). Расстояние делится
         на размер рамки, поэтому один порог годится и для игрока у камеры,
         и для игрока в глубине площадки.

    Владение — состояние, а не признак отдельного кадра. Детектор находит мяч
    примерно в двух кадрах из трёх, поэтому требовать непрерывных попаданий
    нельзя: при чередовании «есть — нет» владелец не успевал подтвердиться, и
    на восьмисекундном фрагменте набиралось меньше секунды владения на обе
    команды — передачи, потери и перехваты не находились вовсе. Поэтому:

      * в окне памяти `window_seconds` считается, сколько времени каждый игрок
        был ближе всех к мячу; владельцем становится тот, у кого набралось
        больше `confirm_seconds` — непрерывная серия кадров не нужна;
      * пропуск короче `bridge_seconds` владение не прерывает: время идёт так
        же, как если бы мяч был виден;
      * если владелец не подтверждался дольше `release_seconds`, он сбрасывается
        (так заканчивается владение при броске и при длинной передаче);
      * при спорной близости (двое почти одинаково близко) владение остаётся у
        прежнего игрока: защитник рядом с ведущим мяч — норма игры, а не повод
        обнулить владение. Если прежнего владельца среди спорных нет, владелец
        неизвестен.
    """

    def __init__(self, confirm_seconds=0.13, release_seconds=0.6, ambiguity_margin=0.05,
                 bridge_seconds=0.35, max_distance=0.9, window_seconds=None, max_step=0.2):
        self.confirm_seconds = confirm_seconds
        self.release_seconds = release_seconds
        self.ambiguity_margin = ambiguity_margin
        self.bridge_seconds = bridge_seconds
        self.max_distance = max_distance
        self.window_seconds = window_seconds or max(3 * confirm_seconds, bridge_seconds)
        self.max_step = max_step
        self.history = deque()  # (время, вклад в секундах, трек или None)
        self.last_hit = {}      # трек → когда он последний раз оказался у мяча
        self.candidate, self.owner = None, None
        self.last_time = None
        self.observed = False  # владение засчитывается в этом кадре

    def _near_ball(self, players, ball):
        """→ [(нормированное расстояние, трек)] по возрастанию, только близкие."""
        bx, by = ball.center
        scored = []
        for p in players:
            if p.track_id is None:
                continue
            x1, y1, x2, y2 = p.box
            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
            distance = float(np.hypot((bx - (x1 + x2) / 2) / w, (by - y1 - 0.55 * h) / h))
            if distance <= self.max_distance:
                scored.append((distance, p.track_id))
        scored.sort()
        return scored

    def _detect(self, players, ball):
        """Кто ближе всех к мячу в этом кадре; None — непонятно."""
        flagged = [p for p in players if p.action == "possession" and p.track_id is not None]
        if len(flagged) == 1:
            return flagged[0].track_id
        pool = flagged if len(flagged) > 1 else players
        if ball is None:
            # мяча в кадре нет, но детектор отметил нескольких как владеющих:
            # если среди них прежний владелец, владение за ним и остаётся
            if len(flagged) > 1 and self.owner in [p.track_id for p in flagged]:
                return self.owner
            return None
        scored = self._near_ball(pool, ball)
        if not scored:
            return None
        if len(scored) > 1 and scored[1][0] - scored[0][0] <= self.ambiguity_margin:
            tied = [track for distance, track in scored
                    if distance - scored[0][0] <= self.ambiguity_margin]
            return self.owner if self.owner in tied else None
        return scored[0][1]

    def _support(self, t):
        """Трек → сколько секунд в окне памяти он был ближе всех к мячу."""
        total = {}
        for moment, weight, track in self.history:
            if track is not None and t - moment <= self.window_seconds:
                total[track] = total.get(track, 0.0) + weight
        return total

    def update(self, t, players, ball):
        step = 0.0 if self.last_time is None else min(self.max_step, max(0.0, t - self.last_time))
        self.last_time = t
        self.candidate = self._detect(players, ball)
        self.history.append((t, step, self.candidate))
        while self.history and t - self.history[0][0] > self.window_seconds:
            self.history.popleft()
        if self.candidate is not None:
            self.last_hit[self.candidate] = t

        support = self._support(t)
        if support:
            # надбавка прежнему владельцу: мяч не переходит из-за пары кадров
            bonus = self.confirm_seconds / 2
            leader = max(support, key=lambda track: support[track] + (bonus if track == self.owner else 0.0))
            if support[leader] >= self.confirm_seconds - 1e-9:
                self.owner = leader
        since_owner = t - self.last_hit.get(self.owner, -1e9)
        if self.owner is not None and since_owner > self.release_seconds:
            self.owner = None
        self.observed = self.owner is not None and since_owner <= self.bridge_seconds
        return self.owner


@dataclass
class Motion:
    """Скорость и дистанция одного трека в метрах.

    * координаты сглаживаются скользящим средним по `window` наблюдениям;
    * скорость меряется по интервалам ≈ interval секунд, а не между соседними
      кадрами (иначе дрожание рамки превращается в «бег»);
    * смещения меньше dead_zone за интервал считаются стоянием на месте;
    * скорость больше max_speed (м/с) — выброс, интервал отбрасывается;
    * пропуск наблюдений дольше max_gap разрывает траекторию: путь через
      невидимый участок не засчитывается.
    """

    interval: float = 0.5
    max_speed: float = 12.0
    max_gap: float = 0.7
    window: int = 5
    dead_zone: float = 0.08
    samples: deque = field(default_factory=deque)
    anchor: tuple | None = None
    last_time: float | None = None
    distance: float = 0.0
    seconds: float = 0.0
    speed: float | None = None
    max_measured: float = 0.0
    rejected: int = 0

    def update(self, t, xy):
        if self.samples.maxlen != self.window:
            self.samples = deque(self.samples, maxlen=self.window)
        if xy is None or (self.last_time is not None and t - self.last_time > self.max_gap):
            self.samples.clear()
            self.anchor, self.speed = None, None
        self.last_time = t
        if xy is None:
            return None
        self.samples.append(np.asarray(xy, dtype=float))
        point = np.mean(self.samples, axis=0)
        if self.anchor is None:
            self.anchor = (t, point)
            return None
        dt = t - self.anchor[0]
        if dt < self.interval - 1e-8:
            return self.speed
        step = float(np.linalg.norm(point - self.anchor[1]))
        self.anchor = (t, point)
        if step / dt > self.max_speed:
            self.rejected += 1
            self.speed = None
            self.samples.clear()
            return None
        if step < self.dead_zone:
            step = 0.0
        self.distance += step
        self.seconds += dt
        self.speed = step / dt
        self.max_measured = max(self.max_measured, self.speed)
        return self.speed


class Statistics:
    """Накопление по трекам: видимость, владение, путь, скорость."""

    def __init__(self, cfg, sample_period=0.1):
        self.cfg = cfg
        self.max_gap = max(0.7, 1.5 * sample_period)
        self.players, self.motion, self.last_seen = {}, {}, {}
        self.team_possession = [0.0, 0.0]

    def update(self, t, players, teams, positions, owner, owner_observed):
        speeds = {}
        for p in players:
            track = p.track_id
            row = self.players.setdefault(
                track,
                dict(track_id=track, team=None, visible_s=0.0, possession_s=0.0, first_seen_s=t, last_seen_s=t),
            )
            if teams.get(track) is not None:
                row["team"] = teams[track]
            if track in self.last_seen:
                dt = t - self.last_seen[track]
                if 0 < dt <= self.max_gap:
                    row["visible_s"] += dt
                    if owner == track and owner_observed:
                        row["possession_s"] += dt
                        if row["team"] in (0, 1):
                            self.team_possession[row["team"]] += dt
            row["last_seen_s"] = t
            self.last_seen[track] = t
            motion = self.motion.setdefault(
                track,
                Motion(self.cfg.metrics_interval, self.cfg.max_speed_m_s, max_gap=self.max_gap,
                       window=self.cfg.smoothing_window, dead_zone=self.cfg.dead_zone_m),
            )
            speeds[track] = motion.update(t, positions.get(track))
        return speeds

    def rows(self):
        output = []
        for track, data in sorted(self.players.items()):
            motion = self.motion[track]
            row = dict(data)
            measured = motion.seconds > 0
            row.update(
                distance_m=round(motion.distance, 2) if measured else None,
                avg_speed_kmh=round(motion.distance / motion.seconds * 3.6, 2) if measured else None,
                max_speed_kmh=round(motion.max_measured * 3.6, 2) if measured else None,
                measured_s=round(motion.seconds, 2),
                rejected_motion_samples=motion.rejected,
            )
            for key in ("visible_s", "possession_s", "first_seen_s", "last_seen_s"):
                row[key] = round(row[key], 3)
            output.append(row)
        return output
