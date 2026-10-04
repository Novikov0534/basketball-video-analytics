# Jersey number model

`jersey_resnet.pt` — рабочие веса ResNet34 для чтения номеров на майках.
Файл намеренно не хранится в Git и не входит в ZIP проекта.

Наши готовые веса опубликованы в GitHub Release `models`. В Google Colab
`scripts/bootstrap_colab.py` скачивает их автоматически в
`models/jersey_resnet.pt`. Переобучение при обычном запуске не требуется.

Для локального/альтернативного источника можно использовать
`JERSEY_RESNET_PATH` или `JERSEY_RESNET_URL`.

`jersey_resnet_report.json` содержит метрики обучения и остаётся в репозитории.
