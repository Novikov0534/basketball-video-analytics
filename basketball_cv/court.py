"""Геометрия баскетбольной площадки в метрах.

Система координат (как в roboflow/sports, ветка feat/basketball):
    x — вдоль длинной стороны, от левой лицевой линии (0) до правой (length);
    y — поперёк, от одной боковой линии (0) до другой (width).

Порядок 33 опорных точек `vertices()` совпадает с порядком ключевых точек
модели Roboflow `basketball-court-detection-2`: i-я точка модели — это
i-я вершина здесь. Именно это соответствие позволяет строить гомографию
«кадр → площадка» без ручной разметки.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CourtSpec:
    league: str
    length: float  # длина площадки, м
    width: float  # ширина площадки, м
    three_point_radius: float  # радиус дуги от центра кольца
    three_point_straight: float  # длина прямого участка трёхочковой от лицевой
    sideline_to_three: float  # расстояние от боковой до прямого участка
    paint_width: float
    paint_length: float  # от лицевой до линии штрафного
    rim_from_baseline: float  # от лицевой до центра кольца
    throw_in_line: float  # от лицевой до отметки ввода мяча
    center_circle_radius: float

    # ------------------------------------------------------------------ точки
    def vertices(self) -> np.ndarray:
        """33 опорные точки площадки (м) в порядке модели ключевых точек."""
        L, W = self.length, self.width
        mid = W / 2
        paint_start = (W - self.paint_width) / 2
        paint_end = paint_start + self.paint_width
        side = self.sideline_to_three
        straight = self.three_point_straight
        rim = self.rim_from_baseline
        arc_top = rim + self.three_point_radius
        points = [
            (0, 0),  # 00 угол площадки
            (0, side),  # 01 начало трёхочковой у лицевой
            (0, paint_start),  # 02 угол зоны у лицевой
            (0, paint_end),  # 03 угол зоны у лицевой
            (0, W - side),  # 04 начало трёхочковой у лицевой
            (0, W),  # 05 угол площадки
            (rim, mid),  # 06 центр левого кольца (проекция на пол)
            (straight, side),  # 07 конец прямого участка трёхочковой
            (straight, W - side),  # 08 конец прямого участка трёхочковой
            (self.paint_length, paint_start),  # 09 угол зоны у штрафной
            (self.paint_length, mid),  # 10 центр линии штрафного
            (self.paint_length, paint_end),  # 11 угол зоны у штрафной
            (self.throw_in_line, 0),  # 12 отметка ввода на боковой
            (arc_top, mid),  # 13 вершина трёхочковой дуги
            (self.throw_in_line, W),  # 14 отметка ввода на боковой
            (L / 2, 0),  # 15 центральная линия × боковая
            (L / 2, mid),  # 16 центр площадки
            (L / 2, W),  # 17 центральная линия × боковая
            (L - self.throw_in_line, 0),  # 18
            (L - arc_top, mid),  # 19 вершина правой дуги
            (L - self.throw_in_line, W),  # 20
            (L - self.paint_length, paint_start),  # 21
            (L - self.paint_length, mid),  # 22 центр правой линии штрафного
            (L - self.paint_length, paint_end),  # 23
            (L - straight, side),  # 24
            (L - straight, W - side),  # 25
            (L - rim, mid),  # 26 центр правого кольца
            (L, 0),  # 27
            (L, side),  # 28
            (L, paint_start),  # 29
            (L, paint_end),  # 30
            (L, W - side),  # 31
            (L, W),  # 32
        ]
        return np.array(points, dtype=np.float64)

    def basket_centers(self) -> np.ndarray:
        return np.array(
            [(self.rim_from_baseline, self.width / 2),
             (self.length - self.rim_from_baseline, self.width / 2)]
        )

    # ------------------------------------------------------------- проверки
    def contains(self, xy, margin: float = 0.3) -> bool:
        x, y = xy
        return -margin <= x <= self.length + margin and -margin <= y <= self.width + margin

    def nearest_basket(self, xy) -> int:
        """0 — левое кольцо, 1 — правое."""
        return 0 if xy[0] <= self.length / 2 else 1

    def distance_to_basket(self, xy) -> float:
        center = self.basket_centers()[self.nearest_basket(xy)]
        return float(math.hypot(xy[0] - center[0], xy[1] - center[1]))

    def is_three_point(self, xy) -> bool:
        """Бросок из-за дуги: в углу — за прямым участком, иначе — дальше радиуса."""
        x, y = xy
        from_baseline = x if self.nearest_basket(xy) == 0 else self.length - x
        if from_baseline <= self.three_point_straight:
            return y < self.sideline_to_three or y > self.width - self.sideline_to_three
        return self.distance_to_basket(xy) > self.three_point_radius

    def is_free_throw_spot(self, xy, tolerance: float = 0.9) -> bool:
        """Бросающий стоит у центра линии штрафного (с небольшим допуском за линией)."""
        x, y = xy
        from_baseline = x if self.nearest_basket(xy) == 0 else self.length - x
        return (
            abs(y - self.width / 2) <= tolerance
            and self.paint_length - 0.3 <= from_baseline <= self.paint_length + tolerance
        )

    def shot_points(self, xy) -> int:
        return 3 if self.is_three_point(xy) else 2


# Размеры взяты из пресетов roboflow/sports (сантиметры → метры).
COURTS = {
    "nba": CourtSpec(
        league="nba",
        length=28.65,
        width=15.24,
        three_point_radius=7.24,
        three_point_straight=4.24,
        sideline_to_three=0.91,
        paint_width=4.88,
        paint_length=5.79,
        rim_from_baseline=1.60,
        throw_in_line=8.35,
        center_circle_radius=1.83,
    ),
    "fiba": CourtSpec(
        league="fiba",
        length=28.0,
        width=15.0,
        three_point_radius=6.75,
        # Официально прямой участок идёт до пересечения с дугой: 1.575 + sqrt(6.75² − 6.6²) ≈ 2.99 м
        # (в пресете roboflow/sports указано 3.30 — это неточность, для FIBA берём 2.99).
        three_point_straight=2.99,
        sideline_to_three=0.90,
        paint_width=4.90,
        paint_length=5.80,
        rim_from_baseline=1.575,
        throw_in_line=8.30,
        center_circle_radius=1.80,
    ),
}


def get_court(league: str) -> CourtSpec:
    try:
        return COURTS[league.lower()]
    except KeyError:
        raise ValueError("Стандарт площадки: nba или fiba.") from None
