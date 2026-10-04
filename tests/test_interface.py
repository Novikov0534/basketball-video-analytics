"""Интерфейс и ноутбук Colab: собираются ли они без запуска моделей."""

from pathlib import Path
import ast

import nbformat
from basketball_cv.app import BACKENDS, DETECTOR_MODELS, LEAGUES, build_app

ROOT = Path(__file__).resolve().parents[1]


def test_app_builds():
    app = build_app()
    assert len(app.blocks) > 15
    assert BACKENDS[list(BACKENDS)[0]] == "roboflow"  # автоматический режим по умолчанию
    assert LEAGUES[list(LEAGUES)[0]] == "nba"



def test_main_ui_has_no_manual_correction_or_calibration():
    text = (ROOT / "basketball_cv/app.py").read_text(encoding="utf-8")
    assert "3 · Коррекция" not in text
    assert "Ручная калибровка" not in text
    assert "select_point" not in text

def test_colab_notebook_is_valid_and_has_no_outputs():
    notebook = nbformat.read(ROOT / "Basketball_CV_Colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) == 3
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.outputs == []


def test_notebook_installs_roboflow_environment():
    text = (ROOT / "Basketball_CV_Colab.ipynb").read_text(encoding="utf-8")
    assert "--roboflow" in text and "ROBOFLOW_API_KEY" in text


def test_presentation_defaults():
    from basketball_cv.app import NUMBER_READERS, REDUCERS, TRACKERS
    assert next(iter(TRACKERS.values())) == "sam2"
    assert next(iter(REDUCERS.values())) == "pca"
    assert next(iter(NUMBER_READERS.values())) == "auto"


def test_referee_class_is_supported():
    from basketball_cv.detectors import classify
    assert classify("referee") == ("referee", "")


def test_analysis_and_output_fps_are_separate():
    from basketball_cv.config import Config
    cfg = Config()
    assert cfg.target_fps == 30
    assert cfg.output_fps == 30


def test_video_event_banner_is_ascii_safe():
    from basketball_cv.events import Event
    from basketball_cv.pipeline import event_banner
    event = Event(1, 1.0, "pass", player_id=11, other_player_id=2)
    text = event_banner(event, {11: "#11", 2: "#2"})
    assert text == "PASS | #11 -> #2"
    assert text.isascii()


def test_detector_choices_for_ab_comparison():
    from basketball_cv.config import Config
    assert Config().detector_model_id == "basketball-player-detection-3-ycjdo/13"
    values = list(DETECTOR_MODELS.values())
    assert values[0] == "basketball-player-detection-3-ycjdo/13"
    assert "basketball-player-detection-3-ycjdo/4" in values
    assert "basketball-player-detection-3-ycjdo/18" in values
