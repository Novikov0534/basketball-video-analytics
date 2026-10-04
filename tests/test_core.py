"""Тесты ядра: геометрия, гомография, команды, номера, события, box score
и сквозной прогон синтетического демо против заранее известной «истины»."""

from pathlib import Path
import json
import threading

import numpy as np
import pytest

from basketball_cv.analytics import BallSelector, Motion, Possession, Statistics
from basketball_cv.boxscore import build_boxscore
from basketball_cv.config import Config
from basketball_cv.court import COURTS, get_court
from basketball_cv.detectors import Detection, ProvidedTracks, classify, make_detection
from basketball_cv.evaluation import evaluate, identity_f1, tracking_quality
from basketball_cv.events import EventEngine, ShotTracker
from basketball_cv.geometry import KeypointCourtMapper, SceneCutDetector
from basketball_cv.jersey import prepare_crop
from basketball_cv.identity import (JerseyNumberAssigner, match_clusters_to_teams, parse_roster,
                                    select_number_crops)
from basketball_cv.ocr import ScoreMonitor, tesseract_command
from basketball_cv.perception import CachedPerception, cache_key
from basketball_cv.perception_worker import attach_tracks
from basketball_cv.pipeline import (analyze, fill_missing_event_teams, frame_labels, make_tracker,
                                    video_info)
from basketball_cv.report import read_csv, save_review
from basketball_cv.teams import ColorEmbedder, TeamClusterer, TrackTeamVoter

ROOT = Path(__file__).resolve().parents[1]
NBA = COURTS["nba"]


def player(track=1, box=(100, 100, 140, 200), action="", confidence=0.9):
    return Detection(box, confidence, "player", action, track)


def ball(x=130, y=150):
    return Detection((x - 5, y - 5, x + 5, y + 5), 0.9, "ball")


# ------------------------------------------------------------------ геометрия
def test_court_vertices_match_keypoint_model_layout():
    vertices = NBA.vertices()
    assert vertices.shape == (33, 2)
    assert tuple(vertices[0]) == (0, 0) and tuple(vertices[32]) == (NBA.length, NBA.width)
    assert vertices[6].tolist() == [NBA.rim_from_baseline, NBA.width / 2]  # центр левого кольца
    assert vertices[16].tolist() == [NBA.length / 2, NBA.width / 2]  # центр площадки


@pytest.mark.parametrize("league", ["nba", "fiba"])
def test_three_point_rule(league):
    court = get_court(league)
    corner = (0.5, 0.3)  # угол — за прямым участком линии
    top_of_key = (court.rim_from_baseline + court.three_point_radius + 0.5, court.width / 2)
    inside = (court.paint_length, court.width / 2)
    assert court.is_three_point(corner) and court.shot_points(corner) == 3
    assert court.is_three_point(top_of_key)
    assert not court.is_three_point(inside) and court.shot_points(inside) == 2


def test_keypoint_homography_recovers_metres_and_rejects_bad_frames():
    court = NBA
    scale, offset = 30.0, np.array([40.0, 25.0])
    pixels = np.array([offset + scale * v for v in court.vertices()])
    mapper = KeypointCourtMapper(court)
    assert mapper.update(0.0, np.column_stack([pixels, np.full(33, 0.9)]))
    x, y = mapper.project(offset + scale * np.array([10.0, 7.0]))
    assert x == pytest.approx(10.0, abs=0.05) and y == pytest.approx(7.0, abs=0.05)
    assert mapper.quality()["mean_reprojection_error_m"] < 0.05
    # все точки на одной линии — гомография вырождена, кадр отбрасывается
    degenerate = np.column_stack([pixels, np.zeros(33)])
    degenerate[:6, 2] = 0.9
    mapper.update(5.0, degenerate)  # позже hold_seconds
    assert not mapper.valid


def test_scene_cut_detected_between_different_scenes():
    detector = SceneCutDetector()
    dark = np.zeros((80, 120, 3), np.uint8)
    bright = np.full((80, 120, 3), 240, np.uint8)
    assert detector.update(dark) is False
    assert detector.update(dark) is False
    assert detector.update(bright) is True


# ------------------------------------------------------------------ команды
def test_team_clustering_separates_two_shirt_colours():
    rng = np.random.default_rng(0)
    crops = []
    for _ in range(12):
        crops.append(np.uint8(np.clip(rng.normal([200, 80, 40], 8, (12, 12, 3)), 0, 255)))
    for _ in range(12):
        crops.append(np.uint8(np.clip(rng.normal([40, 90, 220], 8, (12, 12, 3)), 0, 255)))
    clusterer = TeamClusterer(ColorEmbedder())
    assert clusterer.fit(crops)
    labels = [label for label, _ in clusterer.predict(crops)]
    assert len(set(labels[:12])) == 1 and len(set(labels[12:])) == 1
    assert labels[0] != labels[-1]


def test_team_vote_needs_majority():
    voter = TrackTeamVoter(min_votes=3, min_share=0.6)
    voter.add(1, 0, 0.5)
    voter.add(1, 0, 0.5)
    assert voter.team(1) is None
    voter.add(1, 0, 0.5)
    assert voter.team(1) == 0
    for _ in range(6):
        voter.add(2, 0, 0.5)
        voter.add(2, 1, 0.5)
    assert voter.team(2) is None  # голоса разделились поровну


# ------------------------------------------------------------------ номера
def test_number_assigned_only_when_unambiguous_and_repeated():
    assigner = JerseyNumberAssigner(min_votes=3)
    target = player(1, (100, 100, 140, 200))
    other = player(2, (300, 100, 340, 200))
    number = Detection((110, 130, 130, 150), 0.9, "number", text="7")
    for _ in range(3):
        assigner.update([target, other], [number])
    assert assigner.number(1) == "7" and assigner.confirmed() == {1: "7"}
    # номер внутри двух перекрывающихся игроков — пропускаем
    overlapping = JerseyNumberAssigner(min_votes=1)
    for _ in range(3):
        overlapping.update([target, player(3, (95, 95, 145, 205))], [number])
    assert overlapping.number(1) is None



def test_number_weighted_confidence_breaks_weak_tie():
    assigner = JerseyNumberAssigner(min_votes=2, min_share=0.6)
    assigner.vote(8, "7", 0.92)
    assigner.vote(8, "7", 0.88)
    assigner.vote(8, "1", 0.57)
    assigner.vote(8, "1", 0.56)
    assert assigner.number(8) == "7"


def test_full_video_keeps_ten_number_crops():
    cfg = Config()
    cfg.tune_number_rate(0)
    assert cfg.number_crops_per_track == 10

def test_roster_parsing_and_cluster_matching():
    roster = parse_roster("7 Иванов\n#0 - Петров\n11")
    assert roster == {"7": "Иванов", "0": "Петров", "11": ""}
    permutation, score = match_clusters_to_teams({0: {"3", "11"}, 1: {"7", "0"}}, [roster, {"3": "", "11": ""}])
    assert permutation == [1, 0] and score == 4


# ------------------------------------------------------------------ движение
def test_motion_ignores_jitter_but_measures_real_movement():
    motion = Motion(interval=0.5, window=5, dead_zone=0.08)
    rng = np.random.default_rng(1)
    for i in range(60):
        motion.update(i * 0.1, (5 + rng.normal(0, 0.02), 7 + rng.normal(0, 0.02)))
    assert motion.distance < 0.5  # стоит на месте
    moving = Motion(interval=0.5, window=5, dead_zone=0.08)
    for i in range(61):
        moving.update(i * 0.1, (i * 0.1 * 2.0, 0))  # 2 м/с ровно 6 секунд
    assert moving.distance == pytest.approx(12, rel=0.15)
    assert moving.speed == pytest.approx(2.0, rel=0.15)


def test_teleport_is_rejected_and_gap_breaks_path():
    motion = Motion()
    motion.update(0, (0, 0))
    assert motion.update(0.5, (100, 0)) is None and motion.rejected == 1
    motion.update(1.0, (0, 0))
    before = motion.distance
    motion.update(5.0, (25, 0))  # пропуск наблюдений: путь не засчитывается
    assert motion.distance == before


# ------------------------------------------------------------------ владение
def test_possession_uses_detector_class_first():
    possession = Possession()
    players = [player(1, action="possession"), player(2, (300, 100, 340, 200))]
    possession.update(0, players, None)
    assert possession.update(0.25, players, None) == 1


def test_possession_has_hysteresis_and_ambiguity_is_unknown():
    possession = Possession()
    players = [player(1), player(2, (100, 0, 140, 100))]
    assert possession.update(0, players, ball()) is None
    assert possession.update(0.2, players, ball()) == 1 and possession.observed
    assert possession.update(1.0, players, None) is None
    assert Possession().update(0.3, [player(1), player(2)], ball()) is None


# ------------------------------------------------------------------ события
def shot_signals(engine, times, jump=False, basket=False):
    for t in times:
        engine.update(t, [], {}, None, False, None, [], [], {})


def test_shot_tracker_confirms_start_made_and_miss():
    tracker = ShotTracker(fps=15)
    assert tracker.update(0.0, True, False, False) == []
    assert tracker.update(0.07, True, False, False) == [("start", "jump")]
    assert tracker.update(0.5, False, False, True) == []
    assert tracker.update(0.6, False, False, True) == [("made", "jump")]
    tracker = ShotTracker(fps=15)
    tracker.update(0.0, True, False, False)
    tracker.update(0.07, True, False, False)
    assert tracker.update(2.0, False, False, False) == [("missed", "jump")]


def make_engine():
    return EventEngine(NBA, fps=15)


def test_steal_and_turnover_are_distinguished_from_dead_ball():
    engine = make_engine()
    engine.update(0.0, [], {1: 0}, 1, True, None, [], [], {})
    engine.update(0.5, [], {2: 1}, 2, True, None, [], [], {})
    kinds = [(e.kind, e.player_id) for e in engine.events]
    assert kinds == [("turnover", 1), ("steal", 2)]

    slow = make_engine()
    slow.update(0.0, [], {1: 0}, 1, True, None, [], [], {})
    slow.update(6.0, [], {2: 1}, 2, True, None, [], [], {})
    assert [e.kind for e in slow.events] == ["turnover"]
    assert slow.events[0].outcome == "dead_ball"


def test_pass_and_assist_credited_to_passer():
    engine = make_engine()
    positions = {1: (20.0, 7.0), 2: (19.5, 7.62)}
    engine.update(0.0, [], {1: 1, 2: 1}, 1, True, None, [], [], positions)
    engine.update(0.6, [], {1: 1, 2: 1}, 2, True, None, [], [], positions)
    shooter = player(2, action="shot")
    for t in (1.0, 1.1, 1.2):
        engine.update(t, [shooter], {1: 1, 2: 1}, 2, True, None, [], [], positions)
    for t in (1.5, 1.6):
        engine.update(t, [], {1: 1, 2: 1}, None, False, None, [Detection((0, 0, 1, 1), 0.9, "basket")], [], positions)
    kinds = [e.kind for e in engine.events]
    assert kinds == ["pass", "shot", "made", "assist"]
    assert engine.events[1].points == 3 and engine.events[-1].player_id == 1
    assert engine.score[1] == 3


def test_missed_shot_produces_rebound_for_defending_team():
    engine = make_engine()
    positions = {1: (5.0, 7.62), 2: (6.0, 7.0)}
    shooter = player(1, action="shot")
    engine.update(0.0, [shooter], {1: 0, 2: 1}, 1, True, None, [], [], positions)
    for t in (0.1, 0.2):
        engine.update(t, [shooter], {1: 0, 2: 1}, None, False, ball(), [], [], positions)
    for t in np.arange(0.3, 2.2, 0.1):
        engine.update(float(t), [], {1: 0, 2: 1}, None, False, ball(), [], [], positions)
    engine.update(2.5, [], {1: 0, 2: 1}, 2, True, None, [], [], positions)
    kinds = [e.kind for e in engine.events]
    assert kinds == ["shot", "rebound_def"]
    assert engine.events[0].outcome == "missed" and engine.events[0].points == 2


def test_ball_never_seen_is_not_counted_as_miss():
    engine = make_engine()
    shooter = player(1, action="shot")
    engine.update(0.0, [shooter], {1: 0}, None, False, None, [], [], {})
    engine.update(0.07, [shooter], {1: 0}, None, False, None, [], [], {})
    for t in np.arange(0.2, 2.4, 0.1):
        engine.update(float(t), [], {1: 0}, None, False, None, [], [], {})
    assert engine.events[0].outcome == "unknown"


def test_inbound_after_made_basket_is_not_a_turnover():
    engine = make_engine()
    positions = {1: (19.5, 7.62), 2: (18.0, 7.0)}
    shooter = player(1, action="shot")
    for t in (0.0, 0.1, 0.2):
        engine.update(t, [shooter], {1: 1, 2: 0}, 1, True, None, [], [], positions)
    for t in (0.6, 0.7):
        engine.update(t, [], {1: 1, 2: 0}, None, False, None, [Detection((0, 0, 1, 1), 0.9, "basket")], [], positions)
    engine.update(3.0, [], {1: 1, 2: 0}, 2, True, None, [], [], positions)
    assert [e.kind for e in engine.events] == ["shot", "made"]


def test_free_throw_is_worth_one_point():
    engine = make_engine()
    spot = (NBA.paint_length + 0.2, NBA.width / 2)
    shooter = player(1, action="shot")
    positions = {1: spot, 2: (20.0, 3.0)}
    engine.update(0.0, [shooter], {1: 0, 2: 1}, 1, True, None, [], [], positions)
    engine.update(0.1, [shooter], {1: 0, 2: 1}, 1, True, None, [], [], positions)
    assert engine.events[0].points == 1


def test_geometric_shot_uses_saved_release_position_for_three_pointer():
    """Игрок не должен становиться двухочковым только потому, что успел
    переместиться к кольцу, пока мяч летел.
    """
    engine = EventEngine(NBA, fps=15, has_actions=False)
    hoop = [(600, 200, 660, 220)]
    teams = {11: 0}
    release = {11: (9.0, NBA.width / 2)}  # 7.4 м от левого кольца: 3 очка

    engine.update(0.00, [player(11)], teams, 11, True, ball(300, 300), [], hoop, release)
    engine.update(0.10, [player(11)], teams, 11, True, ball(305, 295), [], hoop, release)
    # Владение потеряно; игрок позже оказался почти под кольцом.
    moved = {11: (1.9, NBA.width / 2)}
    engine.update(0.45, [player(11)], teams, None, False, ball(420, 260), [], hoop, moved)
    engine.update(0.80, [player(11)], teams, None, False, ball(630, 160), [], hoop, moved)

    shot = engine.events[0]
    assert shot.kind == "shot"
    assert shot.time_s == pytest.approx(0.10)
    assert shot.points == 3
    assert shot.x_m == pytest.approx(9.0)
    assert "точка выпуска сохранена" in shot.reason




def test_geometric_made_does_not_spawn_duplicate_shot_from_same_release():
    engine = EventEngine(NBA, fps=15, has_actions=False)
    hoop = [(600, 200, 660, 220)]
    teams = {11: 0}
    release = {11: (9.0, NBA.width / 2)}
    engine.update(0.00, [player(11)], teams, 11, True, ball(300, 300), [], hoop, release)
    engine.update(0.10, [player(11)], teams, 11, True, ball(305, 295), [], hoop, release)
    engine.update(0.45, [player(11)], teams, None, False, ball(420, 260), [], hoop, release)
    engine.update(0.80, [player(11)], teams, None, False, ball(630, 160), [], hoop, release)
    engine.update(1.20, [], teams, None, False, ball(630, 240), [], hoop, {})
    # Ещё несколько кадров около кольца не должны повторно использовать
    # старый owner_snapshot и создавать второй SHOT/MISS.
    engine.update(1.30, [], teams, None, False, ball(628, 245), [], hoop, {})
    engine.update(1.45, [], teams, None, False, ball(625, 255), [], hoop, {})
    engine.update(2.10, [], teams, None, False, None, [], hoop, {})
    shots = [e for e in engine.events if e.kind == "shot"]
    assert len(shots) == 1
    assert shots[0].outcome == "made"


def test_new_owner_does_not_force_active_shot_to_miss_before_timeout():
    engine = EventEngine(NBA, fps=15)
    shooter = player(1, action="shot")
    positions = {1: (8.9, NBA.width / 2), 2: (7.0, 6.0)}
    teams = {1: 0, 2: 1}
    engine.update(0.00, [shooter], teams, 1, True, ball(), [], [], positions)
    engine.update(0.07, [shooter], teams, 1, True, ball(), [], [], positions)
    assert engine.active_shot is not None
    engine.update(0.80, [player(2)], teams, 2, True, ball(300, 300), [], [], positions)
    assert engine.active_shot is not None
    assert engine.events[0].outcome == "pending"
    engine.update(2.00, [], teams, None, False, ball(320, 320), [], [], positions)
    assert engine.active_shot is None
    assert engine.events[0].outcome in ("missed", "unknown")


def test_three_point_tolerance_handles_small_homography_error_at_arc():
    engine = EventEngine(NBA, fps=15, has_actions=False)
    # На 0.16 м внутри идеальной NBA-дуги: при видео-гомографии это должна
    # считаться пограничной 3PT, а не уверенным 2PT.
    radius = NBA.three_point_radius - 0.16
    point = (NBA.rim_from_baseline + radius, NBA.width / 2)
    points, reason = engine._shot_value(1, 0, point, "geometric", {1: 0}, 0.0)
    assert points == 3
    assert "допуск 0.25 м" in reason


def test_trajectory_can_replace_false_owner_with_clear_shooter():
    engine = EventEngine(NBA, fps=15, has_actions=False)
    hoop = [(580, 190, 620, 210)]
    teams = {11: 0, 30: 1}
    false_pos = (0.2, 12.7)
    true_pos = (NBA.rim_from_baseline + NBA.three_point_radius - 0.12, NBA.width / 2)
    # Ложный владелец далеко слева, настоящий стрелок справа на продолжении
    # траектории мяча к кольцу.
    false_player = player(11, box=(120, 500, 200, 700))
    shooter = player(30, box=(900, 220, 1010, 440))
    positions = {11: false_pos, 30: true_pos}
    engine.update(0.00, [false_player, shooter], teams, 11, True, ball(170, 590), [], hoop, positions)
    engine.update(0.10, [false_player, shooter], teams, 11, True, ball(960, 315), [], hoop, positions)
    # После выпуска траектория последовательно идёт к кольцу.
    engine.update(0.30, [false_player, shooter], teams, None, False, ball(850, 285), [], hoop, positions)
    engine.update(0.50, [false_player, shooter], teams, None, False, ball(740, 255), [], hoop, positions)
    engine.update(0.70, [false_player, shooter], teams, None, False, ball(630, 220), [], hoop, positions)
    shots = [e for e in engine.events if e.kind == "shot"]
    assert len(shots) == 1
    assert shots[0].player_id == 30
    assert shots[0].team == 1
    assert shots[0].points == 3
    assert "экстраполяцией" in shots[0].reason

def test_geometric_made_tolerates_missing_ball_frames():
    """Попадание подтверждается по траектории над/под ободом, даже если
    детектор потерял мяч на нескольких промежуточных кадрах.
    """
    engine = EventEngine(NBA, fps=15, has_actions=False)
    hoop = [(600, 200, 660, 220)]
    teams = {11: 0}
    release = {11: (9.0, NBA.width / 2)}
    engine.update(0.00, [player(11)], teams, 11, True, ball(300, 300), [], hoop, release)
    engine.update(0.10, [player(11)], teams, 11, True, ball(305, 295), [], hoop, release)
    engine.update(0.45, [player(11)], teams, None, False, ball(420, 260), [], hoop, release)
    engine.update(0.80, [player(11)], teams, None, False, ball(630, 160), [], hoop, release)
    engine.update(0.93, [], teams, None, False, None, [], hoop, {})
    engine.update(1.06, [], teams, None, False, None, [], hoop, {})
    engine.update(1.20, [], teams, None, False, ball(630, 240), [], hoop, {})

    assert [e.kind for e in engine.events] == ["shot", "made"]
    assert engine.events[0].outcome == "made"
    assert engine.events[1].points == 3
    assert engine.score[0] == 3


# ------------------------------------------------------------------ box score
def test_boxscore_merges_tracks_of_one_player():
    tracks = [
        dict(track_id=1, team=0, number="7", visible_s=10, possession_s=4, distance_m=50, measured_s=10, max_speed_kmh=18),
        dict(track_id=5, team=0, number="7", visible_s=5, possession_s=1, distance_m=25, measured_s=5, max_speed_kmh=21),
        dict(track_id=2, team=1, number="", visible_s=8, possession_s=2, distance_m=30, measured_s=8, max_speed_kmh=15),
    ]
    events = [
        dict(kind="shot", player_id=1, team=0, points=3, status="needs_review"),
        dict(kind="made", player_id=1, team=0, points=3, status="needs_review"),
        dict(kind="steal", player_id=5, team=0, points=None, status="needs_review"),
        dict(kind="turnover", player_id=2, team=1, points=None, status="rejected"),
    ]
    players, teams, mapping = build_boxscore(tracks, events, ["Синие", "Красные"], [{"7": "Иванов"}, {}])
    merged = next(p for p in players if p["number"] == "7")
    assert merged["name"] == "Иванов" and merged["tracks"] == "1 5"
    assert merged["PTS"] == 3 and merged["3PM"] == 1 and merged["STL"] == 1
    assert merged["distance_m"] == 75 and merged["max_speed_kmh"] == 21
    assert teams[0]["PTS"] == 3 and teams[1]["TOV"] == 0  # отклонённое событие не учитывается
    assert mapping[5] == mapping[1]


# ------------------------------------------------------------------ прочее
def test_original_roboflow_class_names_are_supported():
    assert classify("player-jump-shot") == ("player", "shot")
    assert classify("player-in-possession") == ("player", "possession")
    assert classify("ball-in-basket") == ("basket", "made")
    assert classify("referee") == ("referee", "")
    assert make_detection("spectator", (0, 0, 1, 1), 0.5) is None


def test_ball_selector_rejects_teleporting_candidates():
    selector = BallSelector()
    shape = (720, 1280, 3)
    assert selector.choose([ball(100, 100)], 0.0, shape) is not None
    assert selector.choose([ball(1200, 700)], 0.1, shape) is None


def test_score_monitor_requires_stable_readings():
    monitor = ScoreMonitor()
    for t in (0, 0.5):
        monitor.observe(t, [12, 10])
    assert monitor.score is None
    monitor.observe(1, [12, 10])
    assert monitor.score == [12, 10]
    for t in (2, 2.5, 3):
        monitor.observe(t, [12, 100])
    assert monitor.score == [12, 10]  # невозможный скачок отклонён


def test_event_evaluation_is_one_to_one():
    report = evaluate([{"kind": "shot", "time_s": 1}, {"kind": "shot", "time_s": 1.1}],
                      [{"kind": "shot", "time_s": 1.05}])["shot"]
    assert report["tp"] == 1 and report["fp"] == 1 and report["recall"] == 1


def test_config_migrates_version_01_files(tmp_path):
    legacy = tmp_path / "old.json"
    legacy.write_text(json.dumps(dict(backend="yolo", roboflow_model_id="a/1", court_length=28.6512,
                                      image_points=[[0, 0], [1, 0], [1, 1], [0, 1]],
                                      court_points=[[0, 0], [28, 0], [28, 15], [0, 15]],
                                      automatic_teams=True, jersey_ocr=False)), encoding="utf-8")
    cfg = Config.load(legacy)
    assert cfg.detector_model_id == "a/1" and cfg.league == "nba" and cfg.calibration == "manual"


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_start_rejected(value):
    with pytest.raises(ValueError):
        Config(start_seconds=value).validate()


def test_perception_cache_key_depends_on_frame_settings():
    video = ROOT / "examples/demo.mp4"
    base = Config()
    other = Config(target_fps=5)
    assert cache_key(video, base) != cache_key(video, other)
    assert cache_key(video, base) == cache_key(video, Config())


def test_incomplete_perception_file_is_rejected(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(dict(type="header")) + "\n" + json.dumps(dict(type="frame", frame=0, detections=[])),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="не завершён"):
        CachedPerception(path)


# ------------------------------------------------------------------ сквозной прогон
@pytest.fixture(scope="module")
def demo_result(tmp_path_factory):
    cfg = Config.load(ROOT / "examples/demo_config.json")
    detector = CachedPerception(ROOT / "examples/demo_perception.jsonl")
    return analyze(ROOT / "examples/demo.mp4", cfg, tmp_path_factory.mktemp("runs"), detector=detector)


@pytest.fixture(scope="module")
def truth():
    return json.loads((ROOT / "examples/demo_truth.json").read_text(encoding="utf-8"))


def test_demo_detects_every_scripted_event(demo_result, truth):
    folder, summary = demo_result
    events = json.loads((folder / "events.json").read_text(encoding="utf-8"))
    report = evaluate(events, [dict(kind=e["kind"], time_s=e["time_s"]) for e in truth["events"]], tolerance=1.2)
    for kind in {e["kind"] for e in truth["events"]}:
        assert report[kind]["recall"] == 1, (kind, report[kind])
        assert report[kind]["precision"] == 1, (kind, report[kind])


def test_demo_score_teams_and_identities(demo_result, truth):
    folder, summary = demo_result
    assert summary["score"] == truth["score"]
    assert summary["team_names"] == truth["team_names"]
    players = read_csv(folder / "players.csv")
    by_number = {p["number"]: p for p in players}
    assert set(by_number) == set(truth["numbers"].values())
    assert by_number["11"]["name"] == "Сидоров" and by_number["11"]["PTS"] == "3"
    assert by_number["3"]["AST"] == "1" and by_number["3"]["STL"] == "1"
    assert by_number["0"]["team_name"] == "Синие"


def test_demo_distances_close_to_ground_truth(demo_result, truth):
    folder, _ = demo_result
    players = {p["number"]: p for p in read_csv(folder / "players.csv")}
    expected = {"7": truth["distance_m"]["A7"], "0": truth["distance_m"]["A0"],
                "3": truth["distance_m"]["B3"], "11": truth["distance_m"]["B11"]}
    for number, reference in expected.items():
        measured = float(players[number]["distance_m"])
        assert measured == pytest.approx(reference, rel=0.2), (number, measured, reference)


def test_demo_outputs_and_calibration_quality(demo_result):
    folder, summary = demo_result
    assert summary["calibrated_frame_fraction"] == 1.0
    assert summary["calibration_quality"]["mean_reprojection_error_m"] < 0.05
    assert video_info(folder / "annotated.mp4")["duration"] == pytest.approx(15, abs=0.2)
    for name in ("players.csv", "teams.csv", "tracks.csv", "tracks_summary.csv", "events.csv",
                 "shot_chart.png", "heatmap.png", "report.html", "results.zip", "config.json"):
        assert (folder / name).exists(), name
    if tesseract_command():
        assert summary["scoreboard_ocr"] == [0, 3]  # сверка со счётом по броскам


def test_review_recomputes_statistics_without_touching_raw_files(demo_result):
    folder, _ = demo_result
    raw = (folder / "events.csv").read_bytes()
    events = read_csv(folder / "events.csv")
    rows = [[e.get(k, "") for k in ["event_id", "time_s", "kind", "player_id", "team", "status", "outcome",
                                    "points", "reason"]] for e in events]
    for row in rows:
        row[5] = "rejected" if row[2] == "made" else "confirmed"
    identities = [[t["track_id"], t["team"], t["number"], "Проверенное имя"]
                  for t in read_csv(folder / "tracks_summary.csv")]
    save_review(folder, identities, rows, None)
    assert (folder / "events.csv").read_bytes() == raw
    reviewed = json.loads((folder / "summary_reviewed.json").read_text(encoding="utf-8"))
    assert reviewed["score"] == [0, 0] and reviewed["reviewed"]  # попадание отклонено — очки пересчитаны
    assert (folder / "report_reviewed.html").exists()


def test_replay_backend_requires_explicit_detector(tmp_path):
    with pytest.raises(ValueError, match="явно переданный"):
        analyze(ROOT / "examples/demo.mp4", Config(backend="replay", max_seconds=1), tmp_path)


def test_cancel_saves_partial_result(tmp_path):
    stop = threading.Event()
    cfg = Config.load(ROOT / "examples/demo_config.json")

    def stop_during_analysis(fraction, message, frame=None):
        if message.startswith("Анализ"):  # не прерываем этап обучения команд
            stop.set()

    _, summary = analyze(ROOT / "examples/demo.mp4", cfg, tmp_path,
                         detector=CachedPerception(ROOT / "examples/demo_perception.jsonl"),
                         cancel=stop, progress=stop_during_analysis)
    assert summary["cancelled"] and 0 < summary["processed_frames"] < 225


def test_statistics_measure_visibility_in_seconds():
    stats = Statistics(Config(), sample_period=1)
    for t in range(4):
        stats.update(t, [player()], {1: 0}, {1: (float(t), 0.0)}, 1, True)
    row = stats.rows()[0]
    assert row["visible_s"] == 3 and row["possession_s"] == 3 and row["distance_m"] is not None


# ------------------------------------------------- исправления версии 0.2.1
def test_ball_gap_does_not_break_possession():
    """Мяч виден примерно в половине кадров трансляции: пропуск до 0.3 с не должен
    сбрасывать владельца, иначе передачи и потери не находятся."""
    selector = BallSelector()
    shape = (720, 1280, 3)
    assert selector.choose([ball(130, 150)], 0.0, shape) is not None
    held = selector.choose([], 0.2, shape)
    assert held is not None and held.center == (130, 150)
    assert selector.choose([], 0.6, shape) is None  # слишком долгий пропуск


def test_possession_survives_missing_ball_frames():
    possession, selector = Possession(), BallSelector()
    players = [player(1), player(2, (600, 100, 640, 200))]
    owner = None
    for index in range(12):
        t = index * 0.1
        detections = [ball()] if index % 3 == 0 else []  # мяч в каждом третьем кадре
        owner = possession.update(t, players, selector.choose(detections, t, (720, 1280, 3)))
    assert owner == 1


def test_short_tracks_are_excluded_from_players_table(demo_result):
    folder, _ = demo_result
    tracks = read_csv(folder / "tracks_summary.csv")
    players = read_csv(folder / "players.csv")
    assert len(players) <= len(tracks)
    for row in players:
        if not row["number"]:
            assert float(row["visible_s"]) >= 0.5


def test_number_crop_budget_depends_on_clip_length():
    assert Config().tune_number_rate(15).number_crops_per_track == 10
    assert Config().tune_number_rate(60).number_crops_per_track == 8
    assert Config().tune_number_rate(0).number_crops_per_track == 10  # всё видео


# ------------------------------------------------------ версия 0.3: скорость
def test_number_crops_are_selected_by_size_and_spread():
    requests = [dict(track_id=1, frame=f, box=[0, 0, 10, 10], area=100 + f) for f in range(0, 30)]
    requests += [dict(track_id=2, frame=5, box=[0, 0, 5, 5], area=25)]
    selected = select_number_crops(requests, per_track=3)
    first = [r for r in selected if r["track_id"] == 1]
    assert len(first) == 3 and len(selected) == 4  # трек 2 даёт один кроп
    assert min(abs(a["frame"] - b["frame"]) for a in first for b in first if a is not b) >= 3
    assert first[0]["area"] >= 120  # берутся самые крупные рамки


def test_keypoints_are_interpolated_between_reference_frames(tmp_path):
    def row(shift):
        return [[float(i + shift), float(i), 0.9] for i in range(33)]

    path = tmp_path / "sparse.jsonl"
    with path.open("w", encoding="utf-8") as sink:
        sink.write(json.dumps(dict(type="header")) + "\n")
        for index, shift in ((0, 0.0), (4, 4.0)):
            sink.write(json.dumps(dict(type="frame", frame=index, detections=[], keypoints=row(shift))) + "\n")
        for index in (1, 2, 3):
            sink.write(json.dumps(dict(type="frame", frame=index, detections=[], keypoints=None)) + "\n")
        sink.write(json.dumps(dict(type="end")) + "\n")
    perception = CachedPerception(path)
    assert perception.provides_keypoints
    middle = perception.keypoints(frame_index=2)
    assert middle[0][0] == pytest.approx(2.0)  # ровно посередине между 0 и 4
    assert perception.keypoints(frame_index=4)[0][0] == pytest.approx(4.0)


def test_labels_use_number_and_name_when_player_identified():
    tracks = [dict(track_id=1, team=0, number="7", visible_s=5, possession_s=0,
                   distance_m=10, measured_s=5, max_speed_kmh=12),
              dict(track_id=2, team=None, number="", visible_s=5, possession_s=0,
                   distance_m=5, measured_s=5, max_speed_kmh=9)]
    players, _, mapping = build_boxscore(tracks, [], ["Синие", "Красные"], [{"7": "Иванов"}, {}])
    labels = frame_labels(tracks, players, mapping)
    assert labels[1] == "#7 Ivanov"  # кириллица транслитерируется для OpenCV
    assert labels[2] == "id2"


def test_demo_video_scoreboard_follows_made_shots(demo_result):
    folder, summary = demo_result
    assert summary["score_timeline"] == [dict(time_s=7.867, team=1, points=3)]
    assert summary["score"] == [0, 3]
    assert "version" not in summary


# --------------------------------------- версия 0.4: сменный трекер и оценка
def test_tracker_choice_validated_against_backend():
    assert Config(tracker="sam2", backend="roboflow").validate().tracker == "sam2"
    with pytest.raises(ValueError, match="SAM2"):
        Config(tracker="sam2", backend="yolo").validate()
    with pytest.raises(ValueError, match="Трекер"):
        Config(tracker="deepsort").validate()
    with pytest.raises(ValueError, match="размерности"):
        Config(team_reducer="tsne").validate()


def test_provided_tracks_pass_sam2_identifiers_through():
    detections = [
        Detection((0, 0, 10, 20), 0.9, "player", "shot", track_id=4),
        Detection((30, 0, 40, 20), 0.9, "player", "", track_id=None),  # без трека — пропускаем
        Detection((5, 5, 8, 8), 0.9, "ball"),
    ]
    tracked = ProvidedTracks(id_offset=100).update(detections)
    assert [(d.track_id, d.action) for d in tracked] == [(104, "shot")]


def test_pipeline_falls_back_to_bytetrack_when_sam2_absent():
    class WithoutTracks:
        provides_tracks = False

    cfg = Config(tracker="sam2", backend="roboflow")
    assert make_tracker(cfg, WithoutTracks(), fps=15).source == "bytetrack"

    class WithTracks:
        provides_tracks = True

    assert make_tracker(cfg, WithTracks(), fps=15).source == "sam2"


def test_cached_perception_reports_sam2_tracks(tmp_path):
    path = tmp_path / "tracked.jsonl"
    with path.open("w", encoding="utf-8") as sink:
        sink.write(json.dumps(dict(type="header", tracker="sam2")) + "\n")
        sink.write(json.dumps(dict(type="frame", frame=0, detections=[
            dict(label="player", box=[0, 0, 10, 20], conf=0.9, track=7)])) + "\n")
        sink.write(json.dumps(dict(type="end")) + "\n")
    perception = CachedPerception(path)
    assert perception.provides_tracks and perception.tracker == "sam2"
    assert perception.detect(frame_index=0)[0].track_id == 7


def test_tracking_quality_penalises_fragmented_tracks():
    stable = [dict(track_id=t, time_s=i * 0.1, frame=i, x_px=0, y_px=0)
              for t in (1, 2) for i in range(60)]
    torn = [dict(track_id=10 * t + i // 20, time_s=i * 0.1, frame=i, x_px=0, y_px=0)
            for t in (1, 2) for i in range(60)]
    good, bad = tracking_quality(stable, 2), tracking_quality(torn, 2)
    assert good["fragmentation"] == 1.0 and bad["fragmentation"] > good["fragmentation"]
    assert good["mean_track_s"] > bad["mean_track_s"]


def test_identity_f1_counts_switches():
    points = [dict(track_id=1, time_s=0.0, frame=0, x_px=10, y_px=10),
              dict(track_id=2, time_s=1.0, frame=15, x_px=10, y_px=10)]
    reference = [dict(time_s=0.0, player="A", x_px=10, y_px=10),
                 dict(time_s=1.0, player="A", x_px=10, y_px=10)]
    report = identity_f1(points, reference)
    assert report["switches"] == 1 and report["idf1"] == 0.5
    assert identity_f1(points, []) is None


def test_umap_reducer_reports_missing_package_clearly(monkeypatch):
    import builtins

    clusterer = TeamClusterer(ColorEmbedder(), reducer="umap")
    original = builtins.__import__

    def without_umap(name, *args, **kwargs):
        if name == "umap":
            raise ImportError("no umap")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_umap)
    with pytest.raises(RuntimeError, match="umap-learn"):
        clusterer.fit([np.zeros((8, 8, 3), np.uint8) for _ in range(10)])


def test_attach_tracks_matches_by_overlap_and_keeps_lost_players():
    detections = [dict(label="player", box=[0, 0, 10, 20], conf=0.9),
                  dict(label="ball", box=[5, 5, 7, 7], conf=0.9)]
    tracks = [(3, [1, 1, 11, 21]), (4, [100, 0, 110, 20])]  # второй трек без детекции
    matched, players = attach_tracks(detections, tracks)
    assert matched == 1 and players == 1
    assert detections[0]["track"] == 3
    added = [d for d in detections if d.get("from_tracker")]
    assert len(added) == 1 and added[0]["track"] == 4


def test_short_track_still_gets_team_in_final_pass():
    voter = TrackTeamVoter(min_votes=3)
    voter.add(7, 1, 0.5)  # один голос: в реальном времени команды ещё нет
    assert voter.team(7) is None
    assert voter.team(7, relaxed=True) == 1  # в итоговой таблице — есть


def test_event_without_team_inherits_it_from_track():
    from basketball_cv.events import Event

    events = [Event(1, 5.0, "made", player_id=18, team=None, points=3),
              Event(2, 6.0, "made", player_id=99, team=None, points=2)]
    fill_missing_event_teams(events, {18: 1})
    assert events[0].team == 1  # очки попадут в счёт команды
    assert events[1].team is None  # трек неизвестен — не выдумываем


def test_points_of_unassigned_shot_are_not_lost_silently(demo_result):
    folder, summary = demo_result
    scored = sum(e["points"] or 0 for e in json.loads((folder / "events.json").read_text(encoding="utf-8"))
                 if e["kind"] == "made")
    assert sum(summary["score"]) + summary["unassigned_points"] == scored


# ------------------------------------------- чтение номеров классификатором
def test_number_crop_is_padded_to_square_without_stretching():
    wide = np.zeros((20, 60, 3), np.uint8)
    wide[:, :] = 200
    prepared = prepare_crop(wide, size=96)
    assert prepared.shape == (3, 96, 96)
    # исходник вписан по ширине, сверху и снизу — поля
    row_means = prepared.mean(axis=(0, 2))
    assert row_means[0] < row_means[48]
    assert prepare_crop(None) is None and prepare_crop(np.zeros((0, 0, 3), np.uint8)) is None


def test_resnet_reader_processes_every_crop(tmp_path):
    """ResNet читает все найденные рамки, а не выборку: он быстрый."""
    class FakeClassifier:
        name = "resnet"

        def __init__(self):
            self.seen = 0

        def predict(self, crops):
            self.seen += len(crops)
            return [("7", 0.9)] * len(crops)

    from basketball_cv.frames import plan_frames
    from basketball_cv.pipeline import read_all_number_crops

    video = ROOT / "examples/demo.mp4"
    _, plan = plan_frames(video, target_fps=15, max_seconds=2)
    requests = [dict(track_id=1, frame=index, box=[10, 10, 40, 40], area=900)
                for index in range(0, 30, 2)]
    classifier = FakeClassifier()
    assigner = JerseyNumberAssigner(min_votes=2)
    read_all_number_crops(video, plan, requests, classifier, assigner)
    assert classifier.seen > 0
    assert assigner.number(1) == "7"


def test_resnet_requested_without_weights_reports_clearly(tmp_path):
    from basketball_cv.pipeline import read_numbers_for_tracks

    cfg = Config(number_reader="resnet", number_weights=str(tmp_path / "нет.pt"))
    with pytest.raises(ValueError, match="веса ResNet"):
        read_numbers_for_tracks(ROOT / "examples/demo.mp4", None, cfg, JerseyNumberAssigner(),
                                [dict(track_id=1, frame=0, box=[0, 0, 10, 10], area=100)],
                                None, None, None, tmp_path)


def test_number_reader_choice_is_validated():
    assert Config(number_reader="smolvlm").validate().number_reader == "smolvlm"
    with pytest.raises(ValueError, match="Чтение номеров"):
        Config(number_reader="tesseract").validate()


def test_duplicate_tracks_on_one_player_are_dropped():
    """SAM2 может вести двух «объектов» по одному человеку после переподачи."""
    from basketball_cv.perception_worker import drop_duplicate_tracks

    boxes = [(1, [100, 100, 140, 200]), (7, [102, 101, 141, 199]), (3, [300, 100, 340, 200])]
    kept = drop_duplicate_tracks(boxes)
    assert [track for track, _ in kept] == [1, 3]  # остаётся более ранний трек


def test_video_mode_prompts_only_where_players_are_uncovered():
    """Промпты добавляются там, где трека нет, а не по расписанию."""
    from basketball_cv.perception_worker import Sam2VideoTracker

    player_boxes = {0: [[0, 0, 10, 20]], 30: [[0, 0, 10, 20], [200, 0, 210, 20]]}
    tracks = {0: [(1, [0, 0, 10, 20])], 30: [(1, [1, 1, 11, 21])]}
    position, missing = Sam2VideoTracker._find_uncovered(player_boxes, tracks, coverage=0.75)
    assert position == 30 and missing == [[200, 0, 210, 20]]
    # когда все игроки покрыты, добавлять нечего
    full = {0: [(1, [0, 0, 10, 20])], 30: [(1, [1, 1, 11, 21]), (2, [201, 0, 211, 20])]}
    assert Sam2VideoTracker._find_uncovered(player_boxes, full, coverage=0.75) == (None, [])


# ------------------------------------------------- исправления версии 0.5.4
def dribble_scene(t, neighbours=4, spacing=36):
    """Ведение мяча в плотной игре: мяч уходит за рамку и поднимается над головой.

    Так выглядит трансляция: рамка игрока обрезана по телу, мяч в ведении
    регулярно оказывается вне неё, а соседние игроки стоят ближе, чем ширина
    рамки, — поэтому «мяч внутри рамки» плохо работает как признак владения.
    """
    x = 400 + 60 * t
    players = [Detection((x, 300, x + 46, 420), 0.9, "player", "", 1)]
    for k in range(1, neighbours + 1):
        shift = spacing * k * (1 if k % 2 else -1)
        players.append(Detection((x + shift, 296 + 4 * k, x + shift + 46, 416 + 4 * k),
                                 0.9, "player", "", 1 + k))
    bx = x + 23 + 30 * np.sin(9 * t)
    by = 360 + 60 * np.sin(13 * t)
    return players, Detection((bx - 6, by - 6, bx + 6, by + 6), 0.8, "ball")


def measure_possession(seconds=3.0, fps=30, ball_every=1):
    possession, selector = Possession(), BallSelector()
    held = 0.0
    for index in range(round(seconds * fps)):
        t = index / fps
        players, ball = dribble_scene(t)
        detections = [ball] if index % ball_every == 0 else []
        owner = possession.update(t, players, selector.choose(detections, t, (1080, 1920, 3)))
        if owner == 1 and possession.observed:
            held += 1 / fps
    return held


def test_possession_survives_dense_play_and_ball_losses():
    """Главная причина, по которой не находились потери и перехваты.

    Прежняя логика требовала, чтобы мяч попадал в рамку владельца и чтобы
    попадания шли без перерыва. В плотной игре с мячом в двух кадрах из трёх
    владение набиралось меньше секунды на весь фрагмент, и конечный автомат
    событий не видел ни одной смены владения. На этой сцене прежняя логика
    давала 1.3-1.5 с из 3, новая — 2.5 с.
    """
    assert measure_possession(ball_every=1) >= 2.2
    assert measure_possession(ball_every=3) >= 2.2  # мяч виден в каждом третьем кадре


def test_possession_is_not_awarded_to_a_ball_in_flight():
    """Мяч далеко от всех — владельца быть не должно, иначе полёт станет владением."""
    possession, selector = Possession(), BallSelector()
    players = [Detection((300 + 60 * k, 500, 346 + 60 * k, 620), 0.9, "player", "", k + 1)
               for k in range(5)]
    owners = []
    for index in range(30):
        t = index / 30
        x, y = 400 + 300 * t, 250 - 120 * t
        ball_in_air = Detection((x - 6, y - 6, x + 6, y + 6), 0.8, "ball")
        owners.append(possession.update(t, players, selector.choose([ball_in_air], t, (1080, 1920, 3))))
    assert set(owners) == {None}


def test_contested_ball_stays_with_the_current_owner():
    """Защитник вплотную не должен обнулять владение — иначе рвётся цепочка событий."""
    possession = Possession()
    handler = Detection((100, 100, 140, 200), 0.9, "player", "", 1)
    for t in (0.0, 0.05, 0.1, 0.15, 0.2):
        possession.update(t, [handler], ball(120, 155))
    assert possession.owner == 1
    contest = Detection((100, 100, 140, 200), 0.9, "player", "", 2)  # ровно такая же рамка
    assert possession.update(0.25, [handler, contest], ball(120, 155)) == 1
    assert possession.observed
