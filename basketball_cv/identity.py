"""Кто есть кто: номер на майке → трек → игрок из состава.

1. Детектор находит рамки класса `number`, SmolVLM2 читает цифры.
2. Номер относится к игроку, если рамка номера почти целиком лежит внутри
   рамки игрока: IoS = площадь пересечения / площадь номера ≥ 0.9
   (Roboflow делает то же самое с масками SAM2; у нас — с рамками).
   Если подходят несколько игроков (перекрытие), чтение пропускается.
3. Номер трека подтверждается голосованием: минимум 3 одинаковых чтения
   и не меньше 60 % всех чтений трека.
4. Пара (команда, номер) + состав команды → имя игрока. Треки одного
   игрока (после перекрытий ByteTrack выдаёт новые ID) объединяются.
"""

from collections import Counter
import itertools
import re

import numpy as np

NUMBER_PATTERN = re.compile(r"^\d{1,2}$")


def intersection_over_smaller(small_box, big_box):
    ax1, ay1, ax2, ay2 = small_box
    bx1, by1, bx2, by2 = big_box
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    return width * height / area


def pair_numbers(players, number_detections, ios_threshold=0.9):
    """→ [(track_id, рамка номера)] только для однозначных совпадений.

    Номер засчитывается игроку, если его рамка почти целиком внутри рамки
    игрока. Если подходят двое (игроки перекрылись) — пропускаем: лучше
    потерять чтение, чем приписать номер чужому треку.
    """
    pairs = []
    for number in number_detections:
        owners = [p for p in players if intersection_over_smaller(number.box, p.box) >= ios_threshold]
        if len(owners) == 1 and owners[0].track_id is not None:
            pairs.append((owners[0].track_id, number))
    return pairs


def _normalise(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones_like(values)
    return (values - low) / (high - low)


def select_number_crops(requests, per_track=10):
    """Из всех рамок номера трека оставляем самые качественные и разнесённые по времени.

    Не читаем каждую рамку подряд: для ResNet и SmolVLM выбираем несколько
    крупных кадров с разными ракурсами. Это ускоряет обработку и делает
    голосование по номеру устойчивее.
    """
    by_track = {}
    for request in requests:
        by_track.setdefault(request["track_id"], []).append(request)
    selected = []
    for track, items in by_track.items():
        areas = _normalise([float(r.get("area", 0.0)) for r in items])
        sharpness = _normalise([float(r.get("sharpness", 0.0)) for r in items])
        contrast = _normalise([float(r.get("contrast", 0.0)) for r in items])
        detector_conf = _normalise([float(r.get("detector_confidence", 0.0)) for r in items])
        ranked = []
        for item, area, sharp, cont, det_conf in zip(items, areas, sharpness, contrast, detector_conf):
            # Резкость важнее одного только размера: крупный, но смазанный номер
            # хуже чуть меньшего, зато отчётливого кропа.
            score = 0.30 * area + 0.40 * sharp + 0.20 * cont + 0.10 * det_conf
            item["quality_score"] = round(float(score), 4)
            ranked.append(item)
        ranked.sort(key=lambda r: (-r.get("quality_score", 0.0), -r.get("area", 0.0)))
        chosen = []
        for item in ranked:
            # не берём почти соседние кадры: один и тот же ракурс прочитается одинаково
            def separated(other):
                if "time_s" in item and "time_s" in other:
                    return abs(float(item["time_s"]) - float(other["time_s"])) >= 0.15
                return abs(item["frame"] - other["frame"]) >= 3

            if all(separated(other) for other in chosen):
                chosen.append(item)
            if len(chosen) >= per_track:
                break
        selected.extend(chosen)
    selected.sort(key=lambda r: r["frame"])
    return selected


class JerseyNumberAssigner:
    def __init__(self, min_votes=3, min_share=0.6, ios_threshold=0.9):
        self.min_votes, self.min_share, self.ios_threshold = min_votes, min_share, ios_threshold
        self.votes = {}
        self.scores = {}

    def update(self, players, number_detections):
        for track, number in pair_numbers(players, number_detections, self.ios_threshold):
            self.vote(track, number.text)

    def vote(self, track_id, text, confidence=1.0):
        text = str(text).strip()
        if NUMBER_PATTERN.fullmatch(text):
            self.votes.setdefault(track_id, Counter())[text] += 1
            weight = max(0.0, min(1.0, float(confidence)))
            self.scores.setdefault(track_id, Counter())[text] += weight

    def number(self, track_id):
        votes = self.votes.get(track_id)
        scores = self.scores.get(track_id)
        if not votes or not scores:
            return None
        value, weighted = scores.most_common(1)[0]
        count = votes[value]
        total_weight = sum(scores.values())
        weighted_share = weighted / total_weight if total_weight > 0 else 0.0
        if count >= self.min_votes and weighted_share >= self.min_share:
            return value
        return None

    def confirmed(self):
        return {tid: n for tid in self.votes if (n := self.number(tid)) is not None}


def parse_roster(text):
    """Строки вида «0 Tatum» или «0 - Jayson Tatum» → {"0": "Tatum"}."""
    roster = {}
    for line in str(text or "").splitlines():
        match = re.match(r"^\s*#?(\d{1,2})\s*[-–—:.,]?\s*(.*\S)?\s*$", line)
        if match:
            roster[match.group(1)] = (match.group(2) or "").strip()
    return roster


def match_clusters_to_teams(cluster_numbers, rosters):
    """Какой кластер какой команде соответствует, по совпадению номеров с составами.

    cluster_numbers: {0: {"7", "0", ...}, 1: {...}} — подтверждённые номера треков кластера.
    rosters: [состав команды 1, состав команды 2] (dict номер → имя).
    Возвращает (перестановка, число совпадений): перестановка[i] — индекс команды
    для кластера i, или (None, 0), если составы не заданы или данных нет.
    """
    if not any(rosters):
        return None, 0
    best, best_score, scores = None, -1, []
    for permutation in itertools.permutations(range(2)):
        score = sum(len(cluster_numbers.get(c, set()) & set(rosters[t])) for c, t in enumerate(permutation))
        scores.append(score)
        if score > best_score:
            best, best_score = list(permutation), score
    if best_score == 0 or len(set(scores)) == 1:
        return None, best_score  # доказательств недостаточно или обе перестановки равны
    return best, best_score
