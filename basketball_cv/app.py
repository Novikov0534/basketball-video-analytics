"""Веб-интерфейс Basketball CV.

Пользовательский сценарий намеренно оставлен автоматическим: загрузка видео,
необязательные названия/составы команд и запуск анализа. Ручная разметка
площадки и корректировка событий не входят в основной интерфейс; они относятся
к инструментам оценки качества, а не к работе программы.
"""

from pathlib import Path
import threading
import uuid

import gradio as gr

from .boxscore import PLAYER_FIELDS, TEAM_FIELDS
from .cli import demo_setup
from .config import Config
from .identity import parse_roster
from .pipeline import analyze
from .report import read_csv

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
JOBS = {}

BACKENDS = {
    "Roboflow локально — основной": "roboflow",
    "YOLO — базовый режим": "yolo",
    "Roboflow API — облачный режим": "roboflow_api",
    "Синтетическое демо": "replay",
}
LEAGUES = {"NBA (28.65 × 15.24 м)": "nba", "FIBA (28 × 15 м)": "fiba"}

DETECTOR_MODELS = {
    "RF-DETR Medium /13 — основной": "basketball-player-detection-3-ycjdo/13",
    "RF-DETR NAS /18 — экспериментальный": "basketball-player-detection-3-ycjdo/18",
    "YOLOv11s /4 — старый для сравнения": "basketball-player-detection-3-ycjdo/4",
}
TRACKERS = {
    "SAM2 — основной": "sam2",
    "ByteTrack — быстрый fallback": "bytetrack",
}
REDUCERS = {"PCA — основной": "pca", "UMAP — альтернативный": "umap"}
NUMBER_READERS = {
    "ResNet — основной": "auto",
    "SmolVLM2 — альтернативный": "smolvlm",
}
RESULT_EVENT_COLUMNS = [
    "time_s", "kind", "player_id", "other_player_id", "team",
    "outcome", "points", "confidence", "reason",
]

CSS = """
.gradio-container {max-width:1240px!important}
#hero {padding:22px 26px;border-radius:16px;background:#151a21;color:#fff;margin-bottom:16px}
#hero h1 {color:#fff;font-size:28px;margin:0 0 5px}
#hero p {color:#aeb9c7;margin:0}
.compact-note {font-size:13px;color:#8fa0b4}
"""


def run(video, backend, detector_model, league, profile, weights, start, duration, analysis_fps,
        tracker, reducer, number_reader, team_a, team_b, roster_a, roster_b,
        session_id, progress=gr.Progress()):
    """Запускает полностью автоматический анализ видео."""
    detector = None
    if BACKENDS[backend] == "replay":
        video, cfg, detector = demo_setup()
    else:
        if not video:
            raise gr.Error("Загрузите видео.")
        cfg = Config()
        cfg.backend = BACKENDS[backend]
        if cfg.backend in ("roboflow", "roboflow_api"):
            cfg.detector_model_id = DETECTOR_MODELS[detector_model]
        cfg.league = LEAGUES[league]
        cfg.tracker = TRACKERS[tracker] if cfg.backend == "roboflow" else "bytetrack"
        cfg.team_reducer = REDUCERS[reducer]
        cfg.number_reader = NUMBER_READERS[number_reader]
        cfg.weights = weights
        cfg.read_numbers = True
        cfg.image_size = 960 if profile.startswith("Colab GPU") else 640
        cfg.device = "cpu" if profile == "CPU" else "auto"
        cfg.team_names = [team_a.strip() or "Команда 1", team_b.strip() or "Команда 2"]
        cfg.rosters = [parse_roster(roster_a), parse_roster(roster_b)]

        # Основной продукт не требует ручной разметки и исправления результата.
        cfg.calibration = "auto"
        cfg.swap_teams = False
        cfg.image_points = []
        cfg.court_points = []
        cfg.hoops = []
        cfg.score_rois = {}
        cfg.initial_score = None

        cfg.start_seconds = float(start)
        cfg.max_seconds = float(duration)
        cfg.target_fps = float(analysis_fps)
        cfg.output_fps = 30.0
        cfg.tune_number_rate(cfg.max_seconds)

    cancel = threading.Event()
    JOBS[session_id] = cancel
    try:
        folder, summary = analyze(
            video,
            cfg,
            OUTPUTS,
            detector=detector,
            api_key=None,  # берётся из ROBOFLOW_API_KEY окружения/Colab Secrets
            progress=lambda fraction, message, frame=None: progress(fraction, desc=message),
            cancel=cancel,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from None
    finally:
        JOBS.pop(session_id, None)

    players = read_csv(folder / "players.csv")
    teams = read_csv(folder / "teams.csv")
    events = read_csv(folder / "events.csv")
    score = summary["score"]
    names = summary["team_names"]
    text = (
        f"## {names[0]} {score[0]} : {score[1]} {names[1]}\n"
        f"{summary['processed_seconds']:.0f} с видео · "
        f"{summary['player_count']} игроков · {summary['event_count']} событий"
        + (" · обработка остановлена пользователем" if summary["cancelled"] else "")
    )
    warnings = summary.get("warnings", [])
    detector_used = summary.get("models", {}).get("detector", "—")
    diagnostics = (
        f"**Детектор:** {detector_used} · "
        f"**Трекер:** {summary.get('tracker', '—')} · "
        f"**Команды:** {summary.get('team_reducer', '—')} · "
        f"**Номера:** {summary.get('number_reader', '—')} · "
        f"**Анализ:** {summary.get('analysis_fps', '—')} FPS · "
        f"**Видео:** {summary.get('output_fps', '—')} FPS · "
        f"**Судьи:** до {summary.get('max_referees_seen', 0)} в кадре"
    )
    if warnings:
        diagnostics += "\n\n" + "\n".join("• " + w for w in warnings)

    return (
        str(folder / "annotated.mp4"),
        [[t.get(k, "") for k in TEAM_FIELDS] for t in teams],
        [[p.get(k, "") for k in PLAYER_FIELDS] for p in players],
        [[e.get(k, "") for k in RESULT_EVENT_COLUMNS] for e in events],
        str(folder / "shot_chart.png"),
        str(folder / "heatmap.png"),
        str(folder / "results.zip"),
        text,
        diagnostics,
    )


def cancel_job(session_id):
    if session_id in JOBS:
        JOBS[session_id].set()
        return "Останавливаю после текущего кадра; частичный результат будет сохранён."
    return "Активной обработки нет."


def build_app():
    with gr.Blocks(
        title="Basketball CV",
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css=CSS,
    ) as app:
        gr.HTML(
            '<div id="hero"><h1>Basketball CV</h1>'
            '<p>Автоматизированный анализ баскетбольного матча по видеозаписи.</p></div>'
        )
        session_id = gr.State(value=lambda: uuid.uuid4().hex)
        weights = gr.State("yolo11n.pt")

        with gr.Tab("1 · Анализ"):
            video = gr.Video(label="Видео матча", sources=["upload"], height=330)

            with gr.Accordion("Команды и составы (необязательно)", open=True):
                gr.Markdown(
                    "Состав нужен только для связи автоматически распознанного номера с именем. "
                    "Формат строки: `7 Иванов`."
                )
                with gr.Row():
                    team_a = gr.Textbox("Команда 1", label="Команда 1")
                    team_b = gr.Textbox("Команда 2", label="Команда 2")
                with gr.Row():
                    roster_a = gr.Textbox("", lines=5, label="Состав команды 1")
                    roster_b = gr.Textbox("", lines=5, label="Состав команды 2")

            with gr.Accordion("Дополнительные настройки", open=False):
                with gr.Row():
                    detector_model = gr.Dropdown(
                        list(DETECTOR_MODELS), value=list(DETECTOR_MODELS)[0], label="Модель детектора Roboflow"
                    )
                    tracker = gr.Dropdown(list(TRACKERS), value=list(TRACKERS)[0], label="Трекер")
                    reducer = gr.Dropdown(list(REDUCERS), value=list(REDUCERS)[0], label="Разделение команд")
                    number_reader = gr.Dropdown(list(NUMBER_READERS), value=list(NUMBER_READERS)[0], label="Чтение номеров")
                with gr.Row():
                    league = gr.Dropdown(list(LEAGUES), value=list(LEAGUES)[0], label="Стандарт площадки")
                    analysis_fps = gr.Slider(5, 30, value=30, step=1, label="FPS анализа")
                with gr.Row():
                    start = gr.Number(0, label="Начало, с", minimum=0)
                    duration = gr.Number(0, label="Длительность, с (0 = всё видео)", minimum=0)
                with gr.Row():
                    backend = gr.Dropdown(list(BACKENDS), value=list(BACKENDS)[0], label="Backend")
                    profile = gr.Dropdown(
                        ["Colab GPU (T4 / L4 / A100)", "RTX 3050 · 4 ГБ", "CPU"],
                        value="Colab GPU (T4 / L4 / A100)",
                        label="Профиль устройства",
                    )
                gr.Markdown(
                    "Для проверки качества можно обработать один и тот же ролик детектором "
                    "RF-DETR Medium /13 и старым YOLOv11s /4. Вариант /18 оставлен как экспериментальный. "
                    "Итоговое видео сохраняется с частотой до 30 FPS, тяжёлый анализ по умолчанию "
                    "выполняется на 30 FPS."
                )

            with gr.Row():
                run_button = gr.Button("Анализировать матч", variant="primary", size="lg")
                stop_button = gr.Button("Остановить и сохранить")
            status = gr.Markdown("Готов к запуску.")

        with gr.Tab("2 · Результаты"):
            output_video = gr.Video(label="Обработанное видео")
            team_table = gr.Dataframe(headers=TEAM_FIELDS, interactive=False, label="Статистика команд")
            player_table = gr.Dataframe(headers=PLAYER_FIELDS, interactive=False, label="Статистика игроков")
            event_table = gr.Dataframe(headers=RESULT_EVENT_COLUMNS, interactive=False, label="Автоматически найденные события")
            with gr.Row():
                shot_chart = gr.Image(label="Карта бросков")
                heatmap = gr.Image(label="Тепловая карта позиций")
            download = gr.File(label="Скачать результаты (видео, CSV, JSON, HTML)")
            with gr.Accordion("Диагностика", open=False):
                diagnostics = gr.Markdown("Техническая информация появится после обработки.")

        run_button.click(
            run,
            [video, backend, detector_model, league, profile, weights, start, duration, analysis_fps,
             tracker, reducer, number_reader, team_a, team_b, roster_a, roster_b, session_id],
            [output_video, team_table, player_table, event_table, shot_chart, heatmap,
             download, status, diagnostics],
            concurrency_limit=1,
        )
        stop_button.click(cancel_job, [session_id], [status], queue=False)

    return app.queue(max_size=4)
