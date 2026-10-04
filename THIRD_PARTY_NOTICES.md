# Сторонние компоненты и источники

Новый код проекта: AGPL-3.0-only, полный текст в `LICENSE`. Веса, записи матчей и API не включены в эту лицензию автоматически. В архив не включены веса YOLO или приватные данные.

Основой архитектурного разбора служили:

- [Roboflow basketball notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb).
- [Roboflow: How to Identify Basketball Players](https://blog.roboflow.com/identify-basketball-players/).
- [Roboflow sports](https://github.com/roboflow/sports).

Здесь не распространяется копия исходного notebook, его видео, SAM2, SmolVLM2 или Roboflow весов. Адаптер обращается к идентификатору модели, указанному в исходном notebook; доступ зависит от прав и условий аккаунта Roboflow.

Компоненты:

| Компонент | Источник / условия |
|---|---|
| Ultralytics YOLO11 | https://github.com/ultralytics/ultralytics — AGPL-3.0 / коммерческая лицензия поставщика |
| Supervision / ByteTrack implementation | https://github.com/roboflow/supervision — MIT |
| PyTorch / torchvision | https://github.com/pytorch/pytorch и https://github.com/pytorch/vision |
| Gradio | https://github.com/gradio-app/gradio — Apache-2.0 |
| OpenCV | https://github.com/opencv/opencv — Apache-2.0 для используемой версии |
| Tesseract | https://github.com/tesseract-ocr/tesseract — Apache-2.0 |
| uv | https://github.com/astral-sh/uv — MIT / Apache-2.0 |
| FFmpeg | https://ffmpeg.org/legal.html — условия зависят от сборки |
| roboflow/sports (геометрия площадки, автомат броска) | https://github.com/roboflow/sports — MIT |
| SAM2 / segment-anything-2-real-time (необязательный трекер, ставится отдельно) | https://github.com/facebookresearch/sam2 и https://github.com/Gy920/segment-anything-2-real-time — Apache-2.0; веса загружаются с серверов Meta |
| umap-learn (необязательно, team_reducer=umap) | https://github.com/lmcinnes/umap — BSD-3-Clause |
| Roboflow inference (ставится в отдельное окружение .venv-rf) | https://github.com/roboflow/inference — Apache-2.0 с отдельными условиями на часть модулей |
| Transformers / SigLIP (google/siglip-base-patch16-224) | https://github.com/huggingface/transformers — Apache-2.0; веса SigLIP — Apache-2.0 |
| Модели Roboflow Universe (детектор, ключевые точки, номера) | доступ по ключу аккаунта Roboflow; веса в архив не входят |

Точные лицензии устанавливаемых зависимостей находятся в их дистрибутивах. Для библиографии диплома укажите версии из `summary.json`, дату обращения и исходные публикации алгоритмов.
