"""Перевод координат кадра в метры площадки.

Основной режим — автоматический: на каждом кадре модель находит ключевые
точки разметки, по ним строится гомография «кадр → площадка» (RANSAC).
Это не накапливает дрейф при панорамировании камеры и не требует кликов.

Запасной режим — ручная калибровка по 4 точкам на первом кадре плюс
компенсация движения камеры оптическим потоком (как в версии 0.1).
"""

import cv2
import numpy as np


def transform(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def homography(image_points, court_points):
    """Гомография по ручным точкам (ровно заданным, без выбросов)."""
    a, b = np.asarray(image_points, np.float32), np.asarray(court_points, np.float32)
    if len(a) < 4 or len(a) != len(b):
        raise ValueError("Нужно минимум 4 пары точек.")
    if abs(cv2.contourArea(cv2.convexHull(a))) < 20 or abs(cv2.contourArea(cv2.convexHull(b))) < 0.1:
        raise ValueError("Точки калибровки лежат на линии или слишком близко.")
    matrix, _ = cv2.findHomography(a, b, 0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e10:
        raise ValueError("Неустойчивая калибровка.")
    if np.linalg.norm(transform(a, matrix) - b, axis=1).max() > 0.5:
        raise ValueError("Ошибка соответствий больше 0.5 м.")
    return matrix


class SceneCutDetector:
    """Склейка монтажа: резкая смена гистограммы яркости между кадрами."""

    def __init__(self, threshold=0.62):
        self.threshold = threshold
        self.previous = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        cut = False
        if self.previous is not None:
            cut = cv2.compareHist(self.previous, hist, cv2.HISTCMP_BHATTACHARYYA) > self.threshold
        self.previous = hist
        return cut


class KeypointCourtMapper:
    """Гомография на каждом кадре по ключевым точкам площадки.

    Шаги для кадра:
      1. берём точки с уверенностью ≥ min_confidence (нужно ≥ 4);
      2. точки должны охватывать заметную площадь площадки (иначе
         гомография вырождена — например, все точки на одной линии);
      3. cv2.findHomography с RANSAC отбрасывает ошибочные точки;
      4. проверяем среднюю ошибку перепроецирования на инлаерах;
      5. сглаживаем матрицу во времени (экспоненциально), чтобы убрать
         дрожание координат, из-за которого завышалась дистанция;
      6. если кадр не прошёл проверки, до hold_seconds используем
         последнюю хорошую матрицу, потом метры отключаются.
    """

    def __init__(self, court, min_confidence=0.5, max_error_m=0.5, min_area_m2=6.0,
                 hold_seconds=0.7, smoothing=0.6):
        self.court = court
        self.vertices = court.vertices().astype(np.float32)
        self.min_confidence = min_confidence
        self.max_error_m = max_error_m
        self.min_area_m2 = min_area_m2
        self.hold_seconds = hold_seconds
        self.smoothing = smoothing  # вес новой матрицы
        self.matrix = None
        self.last_good_time = -1e9
        self.valid = False
        self.errors = []  # ошибки перепроецирования принятых кадров, м
        self.rejections = {"мало точек": 0, "вырожденная геометрия": 0, "большая ошибка": 0}
        self.reasons = set()

    def reset(self):
        self.matrix, self.valid, self.last_good_time = None, False, -1e9

    def _estimate(self, keypoints):
        if keypoints is None:
            return None, "мало точек"
        keypoints = np.asarray(keypoints, np.float32)
        mask = keypoints[:, 2] >= self.min_confidence
        if mask.sum() < 4:
            return None, "мало точек"
        image_points = keypoints[mask, :2]
        court_points = self.vertices[: len(keypoints)][mask]
        if abs(cv2.contourArea(cv2.convexHull(court_points))) < self.min_area_m2:
            return None, "вырожденная геометрия"
        matrix, inliers = cv2.findHomography(image_points, court_points, cv2.RANSAC, 0.6)
        if matrix is None or inliers is None or inliers.sum() < 4 or not np.isfinite(matrix).all():
            return None, "вырожденная геометрия"
        inliers = inliers.ravel().astype(bool)
        error = np.linalg.norm(transform(image_points[inliers], matrix) - court_points[inliers], axis=1).mean()
        if error > self.max_error_m or np.linalg.cond(matrix) > 1e10:
            return None, "большая ошибка"
        return matrix / matrix[2, 2], float(error)

    def update(self, t, keypoints=None, cut=False, **_):
        if cut:
            self.reset()
        matrix, info = self._estimate(keypoints)
        if matrix is not None:
            if self.matrix is not None and t - self.last_good_time <= self.hold_seconds:
                matrix = self.smoothing * matrix + (1 - self.smoothing) * self.matrix
                matrix = matrix / matrix[2, 2]
            self.matrix, self.last_good_time, self.valid = matrix, t, True
            self.errors.append(info)
        else:
            self.rejections[info] += 1
            self.valid = self.matrix is not None and t - self.last_good_time <= self.hold_seconds
        return self.valid

    def project(self, point):
        if not self.valid:
            return None
        x, y = map(float, transform([point], self.matrix)[0])
        return (x, y) if np.isfinite([x, y]).all() and self.court.contains((x, y)) else None

    def to_image(self, court_xy):
        """Метры → пиксели (для отрисовки дуги, колец и т. п.)."""
        if not self.valid:
            return None
        return transform(court_xy, np.linalg.inv(self.matrix))

    def quality(self):
        return dict(
            mean_reprojection_error_m=round(float(np.mean(self.errors)), 3) if self.errors else None,
            accepted_frames=len(self.errors),
            rejected=dict(self.rejections),
        )


class ManualCourtMapper:
    """Запасной режим: 4 точки на первом кадре + компенсация камеры оптическим потоком."""

    def __init__(self, cfg, court, frame):
        self.cfg = cfg
        self.court = court
        self.h, self.w = frame.shape[:2]
        self.base = homography(np.asarray(cfg.image_points) * [self.w, self.h], cfg.court_points)
        self.current_to_reference = np.eye(3)
        self.previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.valid = True
        self.broken = False
        self.reasons = set()
        self.errors = []

    def update(self, t, keypoints=None, cut=False, frame=None, boxes=()):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cut:
            self.broken = True
            self.reasons.add("Смена сцены: ручная калибровка больше не действует.")
        if self.cfg.camera_mode == "moving" and not self.broken:
            if not self._compensate(gray, boxes):
                self.broken = True
                self.reasons.add("Потеря компенсации движения камеры: последующие метры отключены.")
        self.valid = not self.broken
        self.previous = gray
        return self.valid

    def _compensate(self, gray, boxes):
        mask = np.full(gray.shape, 255, np.uint8)
        mask[: int(self.h * 0.12)] = 0  # верхняя полоса: табло и зрители
        for x1, y1, x2, y2 in boxes:  # игроки движутся сами, их точки не годятся
            cv2.rectangle(mask, (max(0, int(x1) - 5), max(0, int(y1) - 5)), (int(x2) + 5, int(y2) + 5), 0, -1)
        points = cv2.goodFeaturesToTrack(self.previous, 400, 0.02, 10, mask=mask)
        if points is None or len(points) < 16:
            return False
        moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None)
        if moved is None:
            return False
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None)
        if back is None:
            return False
        good = (status.ravel() == 1) & (back_status.ravel() == 1) & (np.linalg.norm(back - points, axis=2).ravel() < 1.5)
        if good.sum() < 12:
            return False
        motion, inliers = cv2.findHomography(points[good], moved[good], cv2.RANSAC, 2.5)
        if motion is None or inliers is None:
            return False
        corners = transform([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]], motion)
        area = abs(cv2.contourArea(corners)) / (self.w * self.h)
        if not (inliers.sum() >= 12 and inliers.mean() >= 0.55 and 0.65 < area < 1.5 and np.linalg.cond(motion) < 1e8):
            return False
        self.current_to_reference = self.current_to_reference @ np.linalg.inv(motion)
        self.current_to_reference /= self.current_to_reference[2, 2]
        return True

    @property
    def matrix(self):
        return self.base @ self.current_to_reference

    def project(self, point):
        if not self.valid:
            return None
        x, y = map(float, transform([point], self.matrix)[0])
        return (x, y) if np.isfinite([x, y]).all() and self.court.contains((x, y)) else None

    def to_image(self, court_xy):
        return transform(court_xy, np.linalg.inv(self.matrix)) if self.valid else None

    def quality(self):
        return dict(mode="manual", broken=self.broken)


class NoCourtMapper:
    """Калибровки нет: метры не вычисляются, остальное работает."""

    valid = False
    reasons = set()

    def update(self, *args, **kwargs):
        return False

    def project(self, point):
        return None

    def to_image(self, court_xy):
        return None

    def quality(self):
        return dict(mode="none")
