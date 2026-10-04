"""Чтение табло (необязательно): используется только для сверки со счётом,
посчитанным по броскам. Номера игроков читает модель Roboflow (identity.py)."""

from collections import Counter
from pathlib import Path
import os
import re
import shutil
import cv2
import numpy as np


def tesseract_command():
    explicit = os.getenv("TESSERACT_CMD")
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("tesseract")
    candidate = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    return found or (str(candidate) if candidate.is_file() else None)


def digits(crop, maximum=199):
    if crop is None or crop.size == 0 or not tesseract_command():
        return None
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = tesseract_command()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    factor = min(6, max(2, 80 / max(1, gray.shape[0])))
    gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:
        binary = 255 - binary
    # обрезаем до самих цифр: широкие белые поля Tesseract иногда принимает за лишний символ
    ink = np.column_stack(np.where(binary < 128))
    if ink.size == 0:
        return None
    (top, left), (bottom, right) = ink.min(axis=0), ink.max(axis=0)
    binary = binary[top:bottom + 1, left:right + 1]
    binary = cv2.copyMakeBorder(binary, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    candidates = []
    # разные режимы сегментации Tesseract дают разный результат на мелких цифрах;
    # берём наиболее частый корректный ответ, а не первый попавшийся
    for mode in (6, 8, 13, 7, 10):
        try:
            value = pytesseract.image_to_string(
                binary, config=f"--psm {mode} -c tessedit_char_whitelist=0123456789", timeout=5
            ).strip()
        except (RuntimeError, pytesseract.TesseractError):
            return None
        if re.fullmatch(r"\d{1,3}", value) and int(value) <= maximum:
            candidates.append(value)
    if not candidates:
        return None
    text = Counter(candidates).most_common(1)[0][0]
    return int(text)


class ScoreMonitor:
    def __init__(self, initial=None, confirmations=3):
        self.score = list(initial) if initial is not None else None
        self.confirmations = confirmations
        self.candidate, self.count = None, 0
        self.rows = []
        self.rejected = 0

    def observe(self, t, pair):
        if pair is None or any(v is None or v < 0 or v > 199 for v in pair):
            self.candidate, self.count = None, 0
            return []
        pair = tuple(map(int, pair))
        if pair != self.candidate:
            self.candidate, self.count = pair, 1
        else:
            self.count += 1
        if self.count < self.confirmations or (
            self.score is not None and list(pair) == self.score
        ):
            return []
        changes = []
        if self.score is not None:
            delta = [pair[i] - self.score[i] for i in range(2)]
            if any(d < 0 or d > 3 for d in delta) or sum(d > 0 for d in delta) > 1:
                self.rejected += 1
                return []
            changes = [(i, d) for i, d in enumerate(delta) if d > 0]
        self.score = list(pair)
        self.rows.append(
            dict(
                time_s=round(t, 3),
                score_a=pair[0],
                score_b=pair[1],
                source="ocr_stable",
            )
        )
        return changes
