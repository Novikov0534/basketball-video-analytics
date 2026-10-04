"""Рисование площадки: тепловая карта, карта бросков, мини-карта в видео."""

import cv2
import numpy as np

TEAM_COLORS = [(235, 140, 40), (40, 110, 240)]  # BGR: синяя и оранжевая группа
UNKNOWN_COLOR = (170, 170, 170)
REFEREE_COLOR = (180, 90, 210)
LINE_COLOR = (225, 225, 225)
FLOOR_COLOR = (34, 42, 56)


class CourtCanvas:
    """Площадка в пикселях: метры → пиксели и отрисовка разметки."""

    def __init__(self, court, width_px=880, margin=28):
        self.court = court
        self.margin = margin
        self.scale = (width_px - 2 * margin) / court.length
        self.width = width_px
        self.height = int(round(court.width * self.scale + 2 * margin))

    def to_px(self, xy):
        return (int(round(self.margin + xy[0] * self.scale)), int(round(self.margin + xy[1] * self.scale)))

    def blank(self):
        canvas = np.full((self.height, self.width, 3), FLOOR_COLOR, np.uint8)
        self.draw_lines(canvas)
        return canvas

    def draw_lines(self, canvas, thickness=2):
        c, s = self.court, self.scale
        color = LINE_COLOR
        cv2.rectangle(canvas, self.to_px((0, 0)), self.to_px((c.length, c.width)), color, thickness)
        cv2.line(canvas, self.to_px((c.length / 2, 0)), self.to_px((c.length / 2, c.width)), color, thickness)
        cv2.circle(canvas, self.to_px((c.length / 2, c.width / 2)), int(c.center_circle_radius * s), color, thickness)
        top, bottom = (c.width - c.paint_width) / 2, (c.width + c.paint_width) / 2
        for side in (0, 1):
            def x(value):
                return value if side == 0 else c.length - value

            cv2.rectangle(canvas, self.to_px((x(0), top)), self.to_px((x(c.paint_length), bottom)), color, thickness)
            rim = (x(c.rim_from_baseline), c.width / 2)
            cv2.circle(canvas, self.to_px(rim), max(3, int(0.23 * s)), color, thickness)
            # прямые участки трёхочковой
            for y in (c.sideline_to_three, c.width - c.sideline_to_three):
                cv2.line(canvas, self.to_px((x(0), y)), self.to_px((x(c.three_point_straight), y)), color, thickness)
            # дуга: от точки пересечения с прямым участком до противоположной
            dy = c.width / 2 - c.sideline_to_three
            dx = c.three_point_straight - c.rim_from_baseline
            start = np.degrees(np.arctan2(-dy, dx))
            angles = np.linspace(start, -start, 60)
            direction = 1 if side == 0 else -1
            points = [
                self.to_px((rim[0] + direction * c.three_point_radius * np.cos(np.radians(a)),
                            rim[1] + c.three_point_radius * np.sin(np.radians(a))))
                for a in angles
            ]
            cv2.polylines(canvas, [np.array(points, np.int32)], False, color, thickness, cv2.LINE_AA)
        return canvas


def heatmap_image(court, positions, width_px=880):
    """positions: список (x, y, вес) в метрах."""
    canvas_obj = CourtCanvas(court, width_px)
    canvas = np.full((canvas_obj.height, canvas_obj.width, 3), FLOOR_COLOR, np.uint8)
    if positions:
        grid = np.zeros((canvas_obj.height, canvas_obj.width), np.float32)
        for x, y, weight in positions:
            px, py = canvas_obj.to_px((x, y))
            if 0 <= px < canvas_obj.width and 0 <= py < canvas_obj.height:
                grid[py, px] += weight
        grid = cv2.GaussianBlur(grid, (0, 0), canvas_obj.scale * 0.9)
        if grid.max() > 0:
            heat = np.uint8(255 * grid / grid.max())
            colored = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
            alpha = (heat.astype(np.float32) / 255.0)[..., None] * 0.85
            canvas = (canvas * (1 - alpha) + colored * alpha).astype(np.uint8)
    canvas_obj.draw_lines(canvas)
    return canvas


def shot_chart_image(court, shots, width_px=880):
    """shots: список dict(x_m, y_m, team, made)."""
    canvas_obj = CourtCanvas(court, width_px)
    canvas = canvas_obj.blank()
    radius = max(5, int(0.28 * canvas_obj.scale))
    for shot in shots:
        if shot.get("x_m") is None or shot.get("y_m") is None:
            continue
        center = canvas_obj.to_px((shot["x_m"], shot["y_m"]))
        color = TEAM_COLORS[shot["team"]] if shot.get("team") in (0, 1) else UNKNOWN_COLOR
        if shot.get("made"):
            cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, radius, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            d = radius
            cv2.line(canvas, (center[0] - d, center[1] - d), (center[0] + d, center[1] + d), color, 3, cv2.LINE_AA)
            cv2.line(canvas, (center[0] - d, center[1] + d), (center[0] + d, center[1] - d), color, 3, cv2.LINE_AA)
    return canvas


def minimap(court, player_positions, width_px=300):
    """player_positions: список ((x, y), команда, признак владения)."""
    canvas_obj = CourtCanvas(court, width_px, margin=10)
    canvas = np.full((canvas_obj.height, canvas_obj.width, 3), FLOOR_COLOR, np.uint8)
    canvas_obj.draw_lines(canvas, thickness=1)
    for xy, team, has_ball in player_positions:
        color = TEAM_COLORS[team] if team in (0, 1) else UNKNOWN_COLOR
        center = canvas_obj.to_px(xy)
        cv2.circle(canvas, center, 5, color, -1, cv2.LINE_AA)
        if has_ball:
            cv2.circle(canvas, center, 8, (0, 220, 255), 2, cv2.LINE_AA)
    return canvas


# OpenCV не умеет рисовать кириллицу — для надписей на видео транслитерируем
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya",
}


def ascii_label(text, limit=18):
    output = []
    for char in str(text):
        lower = char.lower()
        replacement = _TRANSLIT.get(lower, char if char.isascii() else "?")
        output.append(replacement.upper() if char.isupper() and replacement else replacement)
    return "".join(output)[:limit]
