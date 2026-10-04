"""One-to-one event matching, independent of the detector and event heuristics."""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(predicted, reference, tolerance=1.0, match_player=False):
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")

    def kind(e):
        return str(e["kind"]).removesuffix("_candidate")

    predicted = [e for e in predicted if e.get("status") != "rejected"]
    output = {}
    for label in sorted({kind(e) for e in predicted + reference}):
        p = [e for e in predicted if kind(e) == label]
        r = [e for e in reference if kind(e) == label]
        cost = np.full((len(p), len(r)), 1e6)
        for i, a in enumerate(p):
            for j, b in enumerate(r):
                dt = abs(float(a["time_s"]) - float(b["time_s"]))
                if dt <= tolerance and (
                    not match_player
                    or str(a.get("player_id")) == str(b.get("player_id"))
                ):
                    cost[i, j] = dt
        rows, cols = linear_sum_assignment(cost)
        tp = int(sum(cost[i, j] < 1e6 for i, j in zip(rows, cols)))
        precision = tp / len(p) if p else None
        recall = tp / len(r) if r else None
        output[label] = dict(
            tp=tp,
            fp=len(p) - tp,
            fn=len(r) - tp,
            precision=precision,
            recall=recall,
            f1=2 * tp / (len(p) + len(r)) if p or r else None,
        )
    return output


def tracking_quality(track_points, expected_players=10):
    """Метрики качества отслеживания, не требующие ручной разметки.

    Полноценные HOTA и IDF1 требуют покадровой разметки личности, которой на
    этапе разработки нет. Эти показатели считаются из tracks.csv и позволяют
    сравнивать трекеры между собой на одних и тех же данных:

    fragmentation  — сколько треков приходится на одного игрока (идеал 1.0);
    mean_track_s   — средняя длительность трека, с;
    coverage       — среднее число одновременно ведомых игроков к ожидаемому;
    short_share    — доля треков короче 1 секунды (обрывки).
    """
    spans, per_frame = {}, {}
    for point in track_points:
        track = int(point["track_id"])
        t = float(point["time_s"])
        low, high = spans.get(track, (t, t))
        spans[track] = (min(low, t), max(high, t))
        per_frame.setdefault(point["frame"], set()).add(track)
    if not spans:
        return dict(tracks=0, fragmentation=None, mean_track_s=None, coverage=None, short_share=None)
    durations = [high - low for low, high in spans.values()]
    simultaneous = [len(tracks) for tracks in per_frame.values()]
    return dict(
        tracks=len(spans),
        fragmentation=round(len(spans) / expected_players, 2),
        mean_track_s=round(float(np.mean(durations)), 2),
        coverage=round(float(np.mean(simultaneous)) / expected_players, 2),
        short_share=round(sum(d < 1.0 for d in durations) / len(durations), 2),
    )


def identity_f1(track_points, reference, time_tolerance=0.2):
    """IDF1 по ручной разметке: reference = [{"time_s", "player", "x_px", "y_px"}].

    Каждой размеченной позиции сопоставляется ближайший трек того же момента,
    затем считается, насколько устойчиво один трек соответствует одному игроку.
    Возвращает None, если разметки нет.
    """
    if not reference:
        return None
    by_time = {}
    for point in track_points:
        by_time.setdefault(round(float(point["time_s"]), 2), []).append(point)
    links = {}
    matched = 0
    for item in reference:
        t = round(float(item["time_s"]), 2)
        candidates = [p for key, points in by_time.items() if abs(key - t) <= time_tolerance for p in points]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda p: np.hypot(float(p["x_px"]) - float(item["x_px"]),
                                                         float(p["y_px"]) - float(item["y_px"])))
        links.setdefault(str(item["player"]), Counter())[int(nearest["track_id"])] += 1
        matched += 1
    if not matched:
        return None
    correct = sum(counts.most_common(1)[0][1] for counts in links.values())
    return dict(
        idf1=round(correct / matched, 3),
        players=len(links),
        switches=sum(len(counts) - 1 for counts in links.values()),
        matched_points=matched,
    )
