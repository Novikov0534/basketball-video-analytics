"""Игровые события.

Броски. Логика перенесена из ShotEventTracker (roboflow/sports, MIT) и
переведена с кадров на секунды:
  * начало броска — классы детектора `player-jump-shot` / `player-layup-dunk`
    держатся несколько кадров подряд;
  * попадание — класс `ball-in-basket` несколько кадров подряд, пока бросок
    активен; промах — за reset_seconds попадания не было;
  * очки — по позиции бросающего на площадке в момент начала броска:
    за дугой 3, иначе 2; штрафной (1) — бросающий у линии штрафного и рядом
    нет соперников.
Для детекторов без классов действий (COCO YOLO) есть геометрический запасной
вариант: полёт мяча к кольцу и пересечение обода сверху вниз.

Смена владения между игроками:
  * своя команда, короткая пауза — передача (и результативная, если
    получивший забил в течение 3 с);
  * чужая команда, пауза ≤ 1.5 с — перехват + потеря;
  * чужая команда, пауза дольше (мяч был «мёртв»: аут, нарушение, фол) —
    потеря без перехвата;
  * после попадания мяч вводит соперник — это не потеря;
  * после промаха — подбор: в нападении (команда бросавшего) или в защите.

Автоматические события сохраняются со статусом ``automatic``. Ручная
разметка используется только отдельно при оценке качества алгоритма.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

EVENT_NAMES = {
    "shot": "Бросок",
    "made": "Попадание",
    "rebound_off": "Подбор в нападении",
    "rebound_def": "Подбор в защите",
    "steal": "Перехват",
    "turnover": "Потеря",
    "pass": "Передача",
    "assist": "Результативная передача",
    "block": "Блок-шот",
    "score_change": "Изменение табло",
    "scene_cut": "Смена сцены",
}


@dataclass
class Event:
    event_id: int
    time_s: float
    kind: str
    player_id: int | None = None
    other_player_id: int | None = None
    team: int | None = None
    confidence: float = 0.0
    status: str = "automatic"
    outcome: str = ""
    points: int | None = None
    x_m: float | None = None
    y_m: float | None = None
    reason: str = ""


class ShotTracker:
    """Состояние одного броска: старт → попадание или промах по таймауту."""

    def __init__(self, fps, reset_seconds=1.7, min_between_starts=0.5, cooldown_after_made=0.5):
        # 3 кадра при 30 FPS у Roboflow ≈ 0.1 с; не меньше 2 кадров анализа
        self.action_frames = max(2, round(0.1 * fps))
        self.basket_frames = max(2, round(0.067 * fps))
        self.reset_seconds = reset_seconds
        self.min_between_starts = min_between_starts
        self.cooldown_after_made = cooldown_after_made
        self.jump_run = self.layup_run = self.basket_run = 0
        self.active = False
        self.shot_type = ""
        self.start_time = None
        self.last_made_time = -1e9

    def cancel(self):
        self.active, self.shot_type, self.start_time = False, "", None
        self.jump_run = self.layup_run = self.basket_run = 0

    def update(self, t, jump, layup, basket):
        """→ список ("start"|"made"|"missed", тип броска)."""
        events = []
        self.jump_run = self.jump_run + 1 if jump else 0
        self.layup_run = self.layup_run + 1 if layup else 0
        self.basket_run = self.basket_run + 1 if basket else 0
        start_jump = self.jump_run == self.action_frames
        start_layup = self.layup_run == self.action_frames
        start = (start_jump or start_layup) and t - self.last_made_time >= self.cooldown_after_made
        if start and self.active:
            if t - self.start_time >= self.min_between_starts:
                events.append(("missed", self.shot_type))  # новый бросок, а попадания не было
                self.cancel()
            else:
                start = False  # то же движение, а не новый бросок
        if start:
            self.active, self.start_time = True, t
            self.shot_type = "jump" if start_jump else "layup"
            self.basket_run = 0
            events.append(("start", self.shot_type))
        if self.active and not start:
            if self.basket_run >= self.basket_frames:
                events.append(("made", self.shot_type))
                self.last_made_time = t
                self.cancel()
            elif t - self.start_time >= self.reset_seconds:
                events.append(("missed", self.shot_type))
                self.cancel()
        return events


class EventEngine:
    def __init__(self, court, fps, has_actions=True, steal_max_gap=1.5, dead_ball_max_gap=10.0,
                 pass_max_gap=2.5, assist_window=3.0, rebound_window=5.0, inbound_window=8.0):
        self.court = court
        self.fps = fps
        self.has_actions = has_actions
        self.steal_max_gap = steal_max_gap
        self.dead_ball_max_gap = dead_ball_max_gap
        self.pass_max_gap = pass_max_gap
        self.assist_window = assist_window
        self.rebound_window = rebound_window
        self.inbound_window = inbound_window
        self.events = []
        self.score = [0, 0]
        self.unassigned_points = 0
        self._reset_state()

    # ------------------------------------------------------------ служебное
    def _reset_state(self):
        self.shots = ShotTracker(self.fps)
        self.active_shot = None  # Event броска, исход которого ещё не известен
        self.shot_ball_frames = 0  # в скольких кадрах броска был виден мяч
        self.shot_frames = 0
        self.block_run = {}
        self.blocked_shot = False
        self.pending_rebound = None  # (событие броска, время окончания)
        self.current_owner, self.current_team, self.owner_last_time = None, None, -1e9
        self.last_pass = None  # (передающий, получивший, команда, время)
        self.last_made_time = -1e9
        self.last_positions = {}  # трек → (время, (x, y))
        self.previous_ball, self.previous_ball_time = None, -1e9
        self.crossing_frames = 0
        # Для геометрического режима без action-классов. Позиция бросающего
        # фиксируется при последнем подтверждённом владении, до полёта мяча.
        self.owner_snapshot = None  # {player_id, team, time, position}
        self.release_candidate = None
        self.rim_history = deque()  # (time, relative_x, relative_y)
        self.active_shot_deadline = None
        # Один физический выпуск не должен порождать несколько SHOT после
        # того, как первый уже завершён как MADE/MISS.
        self.last_consumed_release_time = -1e9
        self.geometric_cooldown_until = -1e9
        # История в пикселях нужна только как консервативная проверка стрелка:
        # если владение ошибочно привязало мяч к другому игроку, но последующая
        # траектория мяча однозначно экстраполируется к рамке настоящего
        # бросающего, используем этот трек вместо ложного владельца.
        self.player_image_history = deque()  # (time, {track: (box, team, court_xy)})
        self.ball_image_history = deque()    # (time, (x, y), confidence)
        self.target_hoop_px = None           # (cx, cy, width, height)

    def add(self, t, kind, **fields):
        event = Event(len(self.events) + 1, round(float(t), 3), kind, **fields)
        self.events.append(event)
        return event

    def reset(self, t):
        """Склейка монтажа: треки и владение начинаются заново."""
        self.add(t, "scene_cut", confidence=1.0, status="observed",
                 reason="Новый фрагмент: идентификаторы треков независимы.")
        self._reset_state()

    def _position(self, track, t, max_age=1.0):
        record = self.last_positions.get(track)
        if record and t - record[0] <= max_age:
            return record[1]
        return None

    # ----------------------------------------------------------- сигналы
    def _rim_signals(self, t, ball, hoops):
        """Возвращает (рядом_с_кольцом, попадание_по_траектории).

        Детектор мяча может пропускать несколько кадров. Поэтому для
        подтверждения попадания используются наблюдения в окне до 0.8 с:
        мяч должен перейти из области над ободом в область под ним, а
        интерполированная траектория — пройти внутри ширины кольца.
        Телевизионное табло в этой логике не используется.
        """
        while self.rim_history and t - self.rim_history[0][0] > 0.9:
            self.rim_history.popleft()

        near_hoop = False
        crossing = False
        if ball is not None and hoops:
            bx, by = map(float, ball.center)
            candidates = []
            for x1, y1, x2, y2 in hoops:
                w, h = max(8.0, float(x2 - x1)), max(6.0, float(y2 - y1))
                cx, rim_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                rx = (bx - cx) / max(1.0, w / 2.0)
                ry = (by - rim_y) / h
                candidates.append((abs(rx) + 0.25 * abs(ry), rx, ry, cx, rim_y, w, h))
            _, rx, ry, cx, rim_y, w, h = min(candidates, key=lambda item: item[0])
            near_hoop = abs(rx) <= 3.0 and -6.0 <= ry <= 4.0
            if near_hoop:
                self.target_hoop_px = (cx, rim_y, w, h)
                # Соседние детекции не обязательны: допускаем 1–3 пропуска.
                for old_t, old_rx, old_ry in reversed(self.rim_history):
                    dt = t - old_t
                    if dt <= 0 or dt > 0.8:
                        continue
                    if old_ry <= -0.20 and ry >= 0.15 and ry > old_ry:
                        alpha = (0.0 - old_ry) / max(1e-6, ry - old_ry)
                        cross_rx = old_rx + alpha * (rx - old_rx)
                        if abs(cross_rx) <= 1.15:
                            crossing = True
                            break
                self.rim_history.append((t, rx, ry))

        if crossing:
            self.crossing_frames = max(self.crossing_frames, self.shots.basket_frames)
        basket = self.crossing_frames > 0
        self.crossing_frames = max(0, self.crossing_frames - 1)
        return near_hoop, basket

    def _remember_owner(self, t, owner, teams):
        """Сохраняет последнюю надёжную позицию владельца мяча."""
        if owner is None:
            return
        # Новый реально подтверждённый владелец означает новый игровой цикл;
        # защитный cooldown от повторного SHOT старого выпуска больше не нужен.
        if t < self.geometric_cooldown_until:
            self.geometric_cooldown_until = -1e9
        self.owner_snapshot = dict(
            player_id=owner,
            team=teams.get(owner),
            time=float(t),
            position=self._position(owner, t),
        )

    def _remember_image_history(self, t, players, teams, positions, ball):
        """Короткая история рамок игроков и надёжных детекций мяча.

        Она не заменяет обычное владение. История используется только при
        подтверждённом полёте к кольцу, чтобы исправить редкий случай, когда
        ложная детекция мяча привязала выпуск к совсем другому игроку.
        """
        snapshot = {}
        for player in players:
            if player.track_id is None:
                continue
            snapshot[player.track_id] = (
                tuple(map(float, player.box)),
                teams.get(player.track_id),
                positions.get(player.track_id),
            )
        self.player_image_history.append((float(t), snapshot))
        while self.player_image_history and t - self.player_image_history[0][0] > 1.6:
            self.player_image_history.popleft()

        # BallSelector помечает удержанное положение confidence=0.3. Для
        # реконструкции траектории берём только реальные детекции выше него.
        if ball is not None and float(getattr(ball, "confidence", 0.0)) > 0.31:
            self.ball_image_history.append((float(t), np.asarray(ball.center, dtype=float), float(ball.confidence)))
        while self.ball_image_history and t - self.ball_image_history[0][0] > 1.6:
            self.ball_image_history.popleft()

    @staticmethod
    def _ball_to_player_score(point, box):
        """Нормированное расстояние от мяча до области рук/груди игрока."""
        x1, y1, x2, y2 = box
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        hand = np.array(((x1 + x2) / 2.0, y1 + 0.42 * h), dtype=float)
        return float(np.hypot((point[0] - hand[0]) / w, (point[1] - hand[1]) / h))

    def _trajectory_shooter(self, confirmation_t, fallback):
        """Консервативно уточняет стрелка по траектории мяча к кольцу.

        Берётся только монотонный хвост траектории, расстояние которого до
        выбранного кольца уменьшается. Это отсекает случайную ложную детекцию
        мяча перед настоящим полётом. Замена владельца разрешается лишь когда
        экстраполированная назад траектория заметно ближе к рамке другого
        игрока. При слабом сигнале всегда остаётся обычный владелец.
        """
        if self.target_hoop_px is None:
            return fallback
        cx, cy, w, h = self.target_hoop_px
        hoop = np.array((cx, cy), dtype=float)
        obs = [(tt, pt) for tt, pt, conf in self.ball_image_history
               if 0.0 <= confirmation_t - tt <= 1.25]
        if len(obs) < 3:
            return fallback

        # Строим последний непрерывный участок, который действительно
        # приближается к кольцу. Ранний выброс обрывает участок, а не портит fit.
        tolerance = max(12.0, 0.35 * max(w, h))
        reverse_segment = [obs[-1]]
        later_dist = float(np.linalg.norm(obs[-1][1] - hoop))
        for item in reversed(obs[:-1]):
            dist = float(np.linalg.norm(item[1] - hoop))
            if dist + tolerance < later_dist:
                break
            if len(reverse_segment) >= 2:
                # reverse_segment = [поздний, более ранний, ...].
                # Сравниваем два направления в прямом времени:
                # item -> earliest и earliest -> next_later.
                earliest = reverse_segment[-1]
                next_later = reverse_segment[-2]
                v1 = earliest[1] - item[1]
                v2 = next_later[1] - earliest[1]
                norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if norm > 1e-6 and float(np.dot(v1, v2) / norm) < 0.25:
                    break
            reverse_segment.append(item)
            later_dist = dist
        segment = list(reversed(reverse_segment))
        if len(segment) < 3 or segment[-1][0] - segment[0][0] < 0.12:
            return fallback

        times = np.array([item[0] for item in segment], dtype=float)
        points = np.vstack([item[1] for item in segment])
        # Линейная экстраполяция на коротком интервале нужна только для выбора
        # ближайшей рамки, не для физической оценки траектории/попадания.
        origin = times.mean()
        tt = times - origin
        try:
            px = np.polyfit(tt, points[:, 0], 1)
            py = np.polyfit(tt, points[:, 1], 1)
        except Exception:
            return fallback

        release_t = float(fallback.get("time", segment[0][0])) if fallback else float(segment[0][0])
        if release_t < times[0] - 0.45 or release_t > times[-1] + 0.05:
            release_t = float(times[0])
        predicted = np.array((np.polyval(px, release_t - origin), np.polyval(py, release_t - origin)))

        if not self.player_image_history:
            return fallback
        frame_t, players = min(self.player_image_history, key=lambda item: abs(item[0] - release_t))
        if abs(frame_t - release_t) > 0.22 or not players:
            return fallback

        ranked = []
        for track, (box, team, court_xy) in players.items():
            score = self._ball_to_player_score(predicted, box)
            ranked.append((score, track, team, court_xy))
        ranked.sort(key=lambda item: item[0])
        best_score, track, team, court_xy = ranked[0]
        fallback_track = fallback.get("player_id") if fallback else None
        fallback_score = next((r[0] for r in ranked if r[1] == fallback_track), 1e9)

        # Не делаем агрессивную переатрибуцию. Нужны и абсолютная близость,
        # и явное преимущество над текущим владельцем.
        if best_score > 1.10:
            return fallback
        if fallback_track is not None and track != fallback_track and best_score + 0.45 >= fallback_score:
            return fallback
        if track == fallback_track:
            return fallback
        return dict(player_id=track, team=team, time=release_t, position=court_xy,
                    source="trajectory")

    def _three_point_with_tolerance(self, position, margin=0.25):
        """3PT с небольшим допуском на ошибку гомографии у линии.

        Погрешность в 20–25 см типична для стопы игрока на далёком плане и
        не должна превращать бросок с линии в уверенный 2PT. Допуск применяется
        только около самой линии, а не расширяет зону на метры.
        """
        if self.court.is_three_point(position):
            return True, False
        x, y = position
        basket = self.court.nearest_basket(position)
        from_baseline = x if basket == 0 else self.court.length - x
        if from_baseline <= self.court.three_point_straight + margin:
            near_corner_line = (y <= self.court.sideline_to_three + margin or
                                y >= self.court.width - self.court.sideline_to_three - margin)
            return bool(near_corner_line), bool(near_corner_line)
        distance = self.court.distance_to_basket(position)
        near_arc = distance >= self.court.three_point_radius - margin
        return bool(near_arc), bool(near_arc)

    def _geometric_shot_update(self, t, teams, owner, owner_observed, ball, near_hoop, basket):
        """Бросок без action-классов: выпуск → полёт → кольцо.

        Потеря владения сама по себе не считается броском. Сначала сохраняется
        позиция последнего владельца; затем ожидается подтверждение, что мяч
        действительно пришёл к кольцу. Событие при этом получает исходное
        время и координаты выпуска, а не положение игрока возле кольца.
        """
        if self.active_shot is not None:
            if basket:
                self._finish_shot(t, made=True)
            elif self.active_shot_deadline is not None and t >= self.active_shot_deadline:
                self._finish_shot(t, made=False)
            return

        # После завершённого броска те же последние данные владельца и мяча
        # ещё несколько кадров остаются в памяти. Не разрешаем им породить
        # второй SHOT для того же физического выпуска.
        if t < self.geometric_cooldown_until:
            return

        # Если мяч снова подтверждён у игрока, неподтверждённый выпуск был
        # ведением или передачей.
        if owner is not None and owner_observed:
            self.release_candidate = None
            return

        snapshot = self.owner_snapshot
        if self.release_candidate is None and snapshot is not None and ball is not None:
            age = t - snapshot["time"]
            if 0.0 < age <= 0.75 and snapshot["time"] > self.last_consumed_release_time + 1e-6:
                self.release_candidate = dict(snapshot)

        candidate = self.release_candidate
        if candidate is None:
            return
        if t - candidate["time"] > 2.4:
            self.release_candidate = None
            return

        if near_hoop:
            candidate = self._trajectory_shooter(t, candidate)
            self._start_geometric_shot(t, candidate, teams)
            self.release_candidate = None
            # После входа в область кольца ждём точку ниже обода, даже если
            # детектор мяча пропустит несколько промежуточных кадров.
            self.active_shot_deadline = t + 1.2
            if basket:
                self._finish_shot(t, made=True)

    def _start_geometric_shot(self, confirmation_t, candidate, teams):
        shooter = candidate.get("player_id")
        team = candidate.get("team")
        if team is None and shooter is not None:
            team = teams.get(shooter)
        position = candidate.get("position")
        # Без action-класса геометрия уверенно различает 2/3 по точке выпуска;
        # штрафной отдельно не угадываем.
        points, explanation = self._shot_value(
            shooter, team, position, "geometric", teams, candidate["time"]
        )
        corrected = candidate.get("source") == "trajectory"
        confidence = 0.68 if corrected else 0.58
        source_note = (
            "стрелок уточнён обратной экстраполяцией подтверждённой траектории мяча; "
            if corrected else
            "стрелок взят из последнего подтверждённого владения; "
        )
        self.active_shot = self.add(
            candidate["time"], "shot", player_id=shooter, team=team, confidence=confidence,
            outcome="pending", points=points,
            x_m=round(position[0], 2) if position else None,
            y_m=round(position[1], 2) if position else None,
            reason=("Бросок: мяч отделился от игрока и полетел к кольцу; "
                    f"{source_note}точка выпуска сохранена заранее; {explanation}."),
        )
        self.shot_frames = self.shot_ball_frames = 0
        self.block_run, self.blocked_shot = {}, False
        self.pending_rebound = None

    def _choose_shooter(self, t, players, ball):
        shooters = [p for p in players if p.action in ("shot", "layup")]
        if shooters:
            if ball is not None and len(shooters) > 1:
                bx, by = ball.center
                return min(shooters, key=lambda p: np.hypot(p.center[0] - bx, p.center[1] - by)).track_id
            return max(shooters, key=lambda p: p.confidence).track_id
        if self.current_owner is not None and t - self.owner_last_time <= 1.5:
            return self.current_owner
        return None

    def _shot_value(self, shooter, team, position, shot_type, teams, t):
        """→ (очки, пояснение). None — позиция бросающего неизвестна."""
        if position is None:
            return None, "позиция бросающего на площадке не определена"
        if shot_type == "jump" and self.court.is_free_throw_spot(position):
            opponents = [
                xy for track, (time, xy) in self.last_positions.items()
                if t - time <= 0.5 and teams.get(track) is not None and team is not None
                and teams.get(track) != team
            ]
            nearest = min((np.hypot(x - position[0], y - position[1]) for x, y in opponents), default=None)
            if nearest is not None and nearest >= 1.8:
                return 1, "у линии штрафного, соперники не ближе 1.8 м — штрафной"
        is_three, used_tolerance = self._three_point_with_tolerance(position)
        distance = self.court.distance_to_basket(position)
        if is_three:
            note = "; учтён допуск 0.25 м к гомографии у линии" if used_tolerance else ""
            return 3, f"из-за дуги ({distance:.1f} м до кольца{note})"
        return 2, f"внутри дуги ({distance:.1f} м до кольца)"

    # ---------------------------------------------------------- основной шаг
    def update(self, t, players, teams, owner, owner_observed, ball, detections, hoops, positions):
        for track, xy in positions.items():
            if xy is not None:
                self.last_positions[track] = (t, xy)

        self._remember_image_history(t, players, teams, positions, ball)
        near_hoop, geometric_basket = self._rim_signals(t, ball, hoops)

        if self.has_actions:
            jump = any(p.action == "shot" for p in players)
            layup = any(p.action == "layup" for p in players)
            detected_basket = any(d.kind == "basket" for d in detections)
            if detected_basket:
                self.crossing_frames = max(self.crossing_frames, self.shots.basket_frames)
            basket = detected_basket or geometric_basket or self.crossing_frames > 0
            for kind, shot_type in self.shots.update(t, jump, layup, basket):
                if kind == "start":
                    self._start_shot(t, players, teams, ball, shot_type)
                elif kind == "made":
                    self._finish_shot(t, made=True)
                else:
                    self._finish_shot(t, made=False)
        else:
            self._geometric_shot_update(t, teams, owner, owner_observed, ball, near_hoop, geometric_basket)

        if self.active_shot is not None:
            self.shot_frames += 1
            self.shot_ball_frames += int(ball is not None)
            self._check_block(t, players, teams)

        if owner is not None and owner_observed:
            self._possession(t, owner, teams)
            self._remember_owner(t, owner, teams)

        if ball is not None:
            self.previous_ball, self.previous_ball_time = np.array(ball.center), t

    # ------------------------------------------------------------- броски
    def _start_shot(self, t, players, teams, ball, shot_type):
        shooter = self._choose_shooter(t, players, ball)
        team = teams.get(shooter) if shooter is not None else None
        if team is None and shooter is not None and shooter == self.current_owner:
            team = self.current_team
        position = self._position(shooter, t) if shooter is not None else None
        points, explanation = self._shot_value(shooter, team, position, shot_type, teams, t)
        source = "класс броска детектора" if self.has_actions else "мяч полетел к кольцу после владения"
        self.active_shot = self.add(
            t, "shot", player_id=shooter, team=team, confidence=0.75 if self.has_actions else 0.45,
            outcome="pending", points=points,
            x_m=round(position[0], 2) if position else None, y_m=round(position[1], 2) if position else None,
            reason=f"{'Бросок в прыжке' if shot_type == 'jump' else 'Проход / данк'}: {source}; {explanation}.",
        )
        self.shot_frames = self.shot_ball_frames = 0
        self.block_run, self.blocked_shot = {}, False
        self.pending_rebound = None

    def _finish_shot(self, t, made):
        shot = self.active_shot
        if shot is None:
            return
        self.active_shot = None
        self.active_shot_deadline = None
        self.release_candidate = None
        self.rim_history.clear()
        self.crossing_frames = 0
        self.target_hoop_px = None
        self.last_consumed_release_time = max(self.last_consumed_release_time, float(shot.time_s))
        # До нового подтверждённого владения старый snapshot не используется.
        if self.owner_snapshot is not None and self.owner_snapshot.get("time", -1e9) <= shot.time_s + 1e-6:
            self.owner_snapshot = None
        self.geometric_cooldown_until = max(self.geometric_cooldown_until, float(t) + 0.8)
        if made:
            shot.outcome = "made"
            points = shot.points if shot.points is not None else 2
            reason = "Мяч в корзине после броска."
            if shot.points is None:
                reason += " Позиция неизвестна — автоматически засчитано 2 очка."
            self.add(t, "made", player_id=shot.player_id, team=shot.team, confidence=0.7, outcome="made",
                     points=points, x_m=shot.x_m, y_m=shot.y_m, reason=reason)
            if shot.team in (0, 1):
                self.score[shot.team] += points
            else:
                self.unassigned_points += points
            self.last_made_time = t
            self._maybe_assist(t, shot)
            self.pending_rebound = None
        else:
            seen = self.shot_ball_frames / max(1, self.shot_frames)
            # мяч почти не был виден — это не доказательство промаха
            shot.outcome = "missed" if seen >= 0.3 else "unknown"
            self.pending_rebound = (shot, t)

    def _maybe_assist(self, t, shot):
        if self.last_pass is None or shot.player_id is None:
            return
        passer, receiver, team, pass_time = self.last_pass
        if receiver == shot.player_id and passer != receiver and shot.time_s - pass_time <= self.assist_window:
            self.add(t, "assist", player_id=passer, other_player_id=receiver, team=team, confidence=0.5,
                     reason=f"Передача за {shot.time_s - pass_time:.1f} с до результативного броска.")

    def _check_block(self, t, players, teams):
        if self.blocked_shot or t - self.active_shot.time_s > 1.0:
            return
        for p in players:
            if p.action != "block":
                self.block_run.pop(p.track_id, None)
                continue
            self.block_run[p.track_id] = self.block_run.get(p.track_id, 0) + 1
            same_team = teams.get(p.track_id) is not None and teams.get(p.track_id) == self.active_shot.team
            if self.block_run[p.track_id] >= 2 and not same_team and p.track_id != self.active_shot.player_id:
                self.add(t, "block", player_id=p.track_id, other_player_id=self.active_shot.player_id,
                         team=teams.get(p.track_id), confidence=0.55, reason="Класс блок-шота во время броска.")
                self.blocked_shot = True
                return

    # ------------------------------------------------------------ владение
    def _possession(self, t, owner, teams):
        team = teams.get(owner)
        previous, previous_team = self.current_owner, self.current_team
        gap = t - self.owner_last_time
        if previous is None or owner == previous:
            self.current_owner, self.current_team, self.owner_last_time = owner, team, t
            if self.pending_rebound is not None and owner == previous:
                self._rebound(t, owner, team)  # бросавший сам подобрал свой мяч
            return

        # Новый владелец сам по себе не доказывает промах. Исход активного
        # броска меняется только при подтверждённом попадании или по таймауту
        # ShotTracker / геометрического окна наблюдения.

        if self.pending_rebound is not None:
            self._rebound(t, owner, team)
        elif self.active_shot is not None:
            pass  # бросок ещё в полёте — не передача и не потеря
        elif team is None or previous_team is None:
            pass  # без команд тип события определить нельзя
        elif team == previous_team:
            if gap <= self.pass_max_gap:
                self.add(t, "pass", player_id=previous, other_player_id=owner, team=team, confidence=0.6,
                         reason=f"Мяч перешёл партнёру за {gap:.1f} с.")
                self.last_pass = (previous, owner, team, t)
        elif t - self.last_made_time <= self.inbound_window:
            pass  # после попадания мяч вводит соперник — это не потеря
        elif gap <= self.steal_max_gap:
            self.add(t, "turnover", player_id=previous, other_player_id=owner, team=previous_team,
                     confidence=0.55, outcome="steal", reason=f"Мяч отобран соперником за {gap:.1f} с.")
            self.add(t, "steal", player_id=owner, other_player_id=previous, team=team, confidence=0.55,
                     reason=f"Смена владения без паузы ({gap:.1f} с) и без броска.")
        elif gap <= self.dead_ball_max_gap:
            self.add(t, "turnover", player_id=previous, other_player_id=owner, team=previous_team,
                     confidence=0.45, outcome="dead_ball",
                     reason=f"Мяч {gap:.1f} с без владельца, затем у соперника: аут, нарушение или фол.")
        self.current_owner, self.current_team, self.owner_last_time = owner, team, t

    def _rebound(self, t, owner, team):
        shot, end_time = self.pending_rebound
        if t - end_time > self.rebound_window:
            self.pending_rebound = None
            return
        self.pending_rebound = None
        if team is None or shot.team is None:
            return
        offensive = team == shot.team
        uncertain = "" if shot.outcome == "missed" else " Исход броска не был подтверждён наблюдением мяча в корзине."
        self.add(t, "rebound_off" if offensive else "rebound_def", player_id=owner, team=team,
                 other_player_id=shot.player_id, confidence=0.55 if not uncertain else 0.35,
                 reason=f"Первое владение после промаха ({t - shot.time_s:.1f} с после броска).{uncertain}")
