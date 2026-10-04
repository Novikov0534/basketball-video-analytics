"""Сравнение конфигураций системы на одном и том же видео.

Прогоняет один фрагмент в нескольких режимах (трекер, метод кластеризации,
детектор) и выводит сравнительную таблицу: качество отслеживания,
идентификация игроков, найденные события, скорость обработки. Таблица
печатается в Markdown — её можно вставить в главу с экспериментами.

Пример:
    .venv/bin/python scripts/compare_configs.py game.mp4 \\
        --seconds 30 --players 10 \\
        --config "ByteTrack+PCA:tracker=bytetrack,team_reducer=pca" \\
        --config "SAM2+PCA:tracker=sam2,team_reducer=pca" \\
        --config "SAM2+UMAP:tracker=sam2,team_reducer=umap"

Тяжёлый инференс кэшируется, поэтому конфигурации с одинаковым восприятием
(различаются только аналитикой) считаются заметно быстрее первой.
"""

from pathlib import Path
import argparse
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_cv.config import Config  # noqa: E402
from basketball_cv.evaluation import identity_f1, tracking_quality  # noqa: E402
from basketball_cv.pipeline import analyze  # noqa: E402
from basketball_cv.report import read_csv  # noqa: E402

COLUMNS = [
    ("name", "Конфигурация"),
    ("tracker", "Трекер"),
    ("reducer", "Кластеризация"),
    ("tracks", "Треков"),
    ("fragmentation", "Треков на игрока"),
    ("mean_track_s", "Средний трек, с"),
    ("short_share", "Обрывков"),
    ("identified", "Опознано игроков"),
    ("events", "Событий"),
    ("score", "Счёт"),
    ("idf1", "IDF1"),
    ("fps", "Кадр/с обработки"),
    ("wall_s", "Время, с"),
]


def parse_config(text, base):
    """«Имя:ключ=значение,ключ=значение» → (имя, Config)."""
    name, _, assignments = text.partition(":")
    cfg = Config.from_dict(json.loads(json.dumps(base)))
    for pair in filter(None, assignments.split(",")):
        key, _, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        if not hasattr(cfg, key):
            raise SystemExit(f"Неизвестный параметр конфигурации: {key}")
        current = getattr(cfg, key)
        if isinstance(current, bool):
            value = value.lower() in ("1", "true", "да", "yes")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(cfg, key, value)
    return name.strip() or "без имени", cfg.validate()


def run_one(name, cfg, video, output, players, reference):
    detector = None
    if cfg.backend == "replay":  # синтетическое демо: восприятие берётся из файла
        from basketball_cv.perception import CachedPerception

        detector = CachedPerception(Path(__file__).resolve().parents[1] / "examples/demo_perception.jsonl")
    started = time.monotonic()
    folder, summary = analyze(video, cfg, output, detector=detector,
                              progress=lambda fraction, message, frame=None: None)
    tracks = read_csv(folder / "tracks.csv")
    quality = tracking_quality(tracks, expected_players=players)
    identity = identity_f1(tracks, reference) if reference else None
    score = summary["score"]
    return dict(
        name=name,
        tracker=summary.get("tracker", "?"),
        reducer=summary.get("team_reducer", "?"),
        tracks=quality["tracks"],
        fragmentation=quality["fragmentation"],
        mean_track_s=quality["mean_track_s"],
        short_share=quality["short_share"],
        identified=summary["identified_tracks"],
        events=summary["event_count"],
        score=f"{score[0]}:{score[1]}",
        idf1=identity["idf1"] if identity else "—",
        fps=round(summary["processed_frames"] / max(0.01, time.monotonic() - started), 2),
        wall_s=round(summary["wall_seconds"], 1),
        folder=str(folder),
        warnings=summary["warnings"],
    )


def markdown_table(rows):
    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    divider = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "—")) for key, _ in COLUMNS) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Сравнение конфигураций на одном видео")
    parser.add_argument("video")
    parser.add_argument("--config", action="append", required=True,
                        help="«Имя:ключ=значение,...»; можно повторять")
    parser.add_argument("--base-config", help="JSON с общими параметрами")
    parser.add_argument("--output", default="outputs/comparison")
    parser.add_argument("--seconds", type=float, help="длительность фрагмента")
    parser.add_argument("--players", type=int, default=10, help="сколько игроков на площадке")
    parser.add_argument("--reference", help="CSV разметки личностей для IDF1")
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY", ""))
    arguments = parser.parse_args()

    base = json.loads(Path(arguments.base_config).read_text(encoding="utf-8")) if arguments.base_config else {}
    if arguments.seconds is not None:
        base["max_seconds"] = arguments.seconds
    reference = read_csv(arguments.reference) if arguments.reference else None
    if arguments.api_key:
        os.environ["ROBOFLOW_API_KEY"] = arguments.api_key

    rows = []
    for text in arguments.config:
        name, cfg = parse_config(text, base)
        print(f"→ {name}: трекер {cfg.tracker}, кластеризация {cfg.team_reducer}", flush=True)
        try:
            rows.append(run_one(name, cfg, arguments.video, arguments.output, arguments.players, reference))
        except Exception as exc:  # одна упавшая конфигурация не отменяет сравнение
            print(f"  не выполнено: {exc}", flush=True)
            rows.append(dict(name=name, tracker="ошибка", reducer=str(exc)[:60]))
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.md").write_text(markdown_table(rows), encoding="utf-8")
    (output / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + markdown_table(rows))
    print("\nСохранено:", output / "comparison.md")


if __name__ == "__main__":
    main()
