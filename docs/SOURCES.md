# Первичные источники

- Исходный [basketball notebook Roboflow](https://github.com/roboflow/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb) и [статья](https://blog.roboflow.com/identify-basketball-players/).
- [YOLO11](https://docs.ultralytics.com/models/yolo11/), [исходный код v8.3.228](https://github.com/ultralytics/ultralytics/tree/v8.3.228).
- [Supervision ByteTrack](https://supervision.roboflow.com/0.27.0/trackers/) и [код](https://github.com/roboflow/supervision/tree/0.27.0).
- [ByteTrack, ECCV 2022, arXiv:2110.06864](https://arxiv.org/abs/2110.06864).
- [PyTorch: предыдущие версии и команды установки](https://pytorch.org/get-started/previous-versions/): torch 2.8.0 / torchvision 0.23.0, CUDA 12.6 или CPU.
- [OpenCV: гомография](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html), [оптический поток](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html).
- [Tesseract: улучшение распознавания](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html).
- [uv: управляемый Python](https://docs.astral.sh/uv/guides/install-python/).
- [Gradio: исходный код](https://github.com/gradio-app/gradio/tree/gradio%405.49.1).
- [Roboflow hosted inference client v0.62.5](https://github.com/roboflow/inference/tree/v0.62.5/inference_sdk): формат HTTP API адаптера.

Размеры калибровки необходимо сверять с разметкой фактической площадки и правилами соревнования. Пресеты проекта — только прямоугольники FIBA 28 × 15 м / 5.8 × 4.9 м; не универсальное допущение для любого матча.

## Добавлено в версии 0.2

- [roboflow/sports, ветка feat/basketball](https://github.com/roboflow/sports/tree/feat/basketball) — MIT: геометрия площадки (`sports/basketball/config.py`, 33 вершины, пресеты NBA/FIBA) и автомат броска `ShotEventTracker` (`sports/basketball/tools.py`), перенесённые в `court.py` и `events.py`.
- Модели Roboflow Universe: `basketball-player-detection-3-ycjdo/13` (RF-DETR Medium, основной детектор, 10 классов, включая `referee`); `basketball-player-detection-3-ycjdo/4` (YOLOv11s, сохранён только для A/B-сравнения); `basketball-player-detection-3-ycjdo/18` (RF-DETR NAS, экспериментальный вариант), `basketball-court-detection-2/14` (33 ключевые точки площадки), `basketball-jersey-numbers-ocr/3` (SmolVLM2 + LoRA).
- [Roboflow inference 1.7.1](https://github.com/roboflow/inference) — локальный запуск моделей Universe.
- [SigLIP, arXiv:2303.15343](https://arxiv.org/abs/2303.15343), веса `google/siglip-base-patch16-224` — эмбеддинги для кластеризации команд.
- [Savitzky–Golay / робастная чистка траекторий](https://github.com/roboflow/sports/blob/feat/basketball/sports/common/path.py) — идея сглаживания путей.
- Официальные размеры площадки: правила NBA и [FIBA Basketball Equipment/Official Basketball Rules](https://www.fiba.basketball/documents) — для проверки пресетов (прямой участок трёхочковой FIBA взят как 2.99 м, а не 3.30 м из пресета Roboflow).

## Добавлено в версии 0.4

- [SAM2 (Segment Anything Model 2), arXiv:2408.00714](https://arxiv.org/abs/2408.00714) — модель сегментации и отслеживания в видео.
- [Gy920/segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time) — форк SAM2 с потоковым интерфейсом (`build_sam2_camera_predictor`), тот же, что в ноутбуке Roboflow. Веса `sam2.1_hiera_small.pt` с dl.fbaipublicfiles.com.
- [UMAP, arXiv:1802.03426](https://arxiv.org/abs/1802.03426) — нелинейное понижение размерности, используется в оригинале для кластеризации команд.
- [HOTA, IJCV 2021, arXiv:2009.07736](https://arxiv.org/abs/2009.07736) и [IDF1, arXiv:1609.01775](https://arxiv.org/abs/1609.01775) — метрики качества отслеживания; в проекте реализован упрощённый IDF1 по собственной разметке, полноценные метрики считаются внешним [TrackEval](https://github.com/JonathonLuiten/TrackEval).

- [umap-learn 0.5.12](https://pypi.org/project/umap-learn/0.5.12/) — альтернативное понижение размерности для сравнения с PCA.
