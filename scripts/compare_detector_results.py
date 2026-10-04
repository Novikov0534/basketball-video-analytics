#!/usr/bin/env python3
"""Сравнение двух запусков Basketball CV с разными детекторами.

Пример:
    python scripts/compare_detector_results.py outputs/run-old outputs/run-new
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    "processed_seconds", "wall_seconds", "track_count", "identified_tracks",
    "player_count", "event_count", "ball_detection_fraction",
    "referee_detection_fraction", "calibrated_frame_fraction", "number_crops_read",
)

def load(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    return json.loads(p.read_text(encoding="utf-8"))

def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнить два результата разных детекторов")
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()
    a, b = load(args.first), load(args.second)
    name_a = a.get("models", {}).get("detector", "first")
    name_b = b.get("models", {}).get("detector", "second")
    width = max(24, len(name_a), len(name_b))
    print(f"{'Метрика':30} | {name_a:{width}} | {name_b:{width}}")
    print("-" * (34 + width * 2))
    for key in METRICS:
        print(f"{key:30} | {str(a.get(key, '—')):{width}} | {str(b.get(key, '—')):{width}}")

if __name__ == "__main__":
    main()
