"""Скачивание датасета номеров с Roboflow без установки их пакета.

Пакет `roboflow` тянет свои версии typer и opencv и может понизить те, что
уже стоят в окружении приложения. Здесь используется только HTTP API:
запрос ссылки на экспорт и загрузка архива.

Датасет `basketball-jersey-numbers-ocr` имеет тип text-image-pairs, поэтому
доступен в форматах jsonl / openai / florence2-od, но не в виде папок по
классам. Скрипт качает jsonl — скрипт обучения его понимает.

    .venv/bin/python scripts/download_jersey_dataset.py --output /content/jersey
"""

from pathlib import Path
import argparse
import io
import json
import os
import time
import zipfile

import requests

API = "https://api.roboflow.com"


def request_export(workspace, project, version, fmt, key, attempts=30, pause=10):
    """Roboflow готовит экспорт не мгновенно: повторяем запрос, пока не будет ссылки."""
    url = f"{API}/{workspace}/{project}/{version}/{fmt}"
    for attempt in range(1, attempts + 1):
        response = requests.get(url, params={"api_key": key}, timeout=60)
        if response.status_code == 200:
            payload = response.json()
            link = payload.get("export", {}).get("link") or payload.get("link")
            if link:
                return link
            print(f"  экспорт готовится… ({attempt}/{attempts})", flush=True)
        elif response.status_code == 202:
            print(f"  экспорт готовится… ({attempt}/{attempts})", flush=True)
        else:
            raise SystemExit(f"Roboflow: HTTP {response.status_code}\n{response.text[:500]}")
        time.sleep(pause)
    raise SystemExit("Roboflow не подготовил экспорт за отведённое время — попробуйте позже.")


def download_archive(link, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with requests.get(link, stream=True, timeout=300) as response:
        response.raise_for_status()
        buffer = io.BytesIO()
        size = 0
        for chunk in response.iter_content(chunk_size=1 << 20):
            buffer.write(chunk)
            size += len(chunk)
            print(f"\r  загружено {size / 1e6:.1f} МБ", end="", flush=True)
    print()
    with zipfile.ZipFile(buffer) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise SystemExit("Некорректный путь в архиве датасета.")
        archive.extractall(destination)
    return destination


def describe(folder):
    """Короткая сводка о том, что скачалось: помогает, если формат неожиданный."""
    jsonl = list(folder.rglob("*.jsonl"))
    images = [p for p in folder.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"\nПапка: {folder}")
    print(f"Файлов JSONL: {len(jsonl)}, изображений: {len(images)}")
    for path in jsonl[:5]:
        lines = path.read_text(encoding="utf-8").splitlines()
        print(f"  {path.relative_to(folder)} — записей: {len(lines)}")
        if lines:
            print(f"    пример: {lines[0][:160]}")
    labels = set()
    for path in jsonl:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                value = str(record.get("suffix") or record.get("label") or record.get("text") or "").strip()
                if value:
                    labels.add(value)
    if labels:
        print(f"Разных номеров: {len(labels)} — например, {sorted(labels)[:12]}")


def main():
    parser = argparse.ArgumentParser(description="Скачивание датасета номеров с Roboflow")
    parser.add_argument("--output", default="datasets/jersey")
    parser.add_argument("--workspace", default="roboflow-jvuqo")
    parser.add_argument("--project", default="basketball-jersey-numbers-ocr")
    parser.add_argument("--version", default="3")
    parser.add_argument("--format", default="jsonl",
                        help="jsonl (по умолчанию), openai или florence2-od")
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY", ""))
    options = parser.parse_args()

    if not options.api_key:
        raise SystemExit("Нужен ROBOFLOW_API_KEY: секрет Colab или переменная окружения.")
    print(f"Запрашиваю {options.project} версии {options.version} в формате {options.format}", flush=True)
    link = request_export(options.workspace, options.project, options.version,
                          options.format, options.api_key)
    folder = download_archive(link, Path(options.output))
    describe(folder)
    print("\nГотово. Обучение:")
    print(f"  python scripts/train_jersey_resnet.py --data {folder}")


if __name__ == "__main__":
    main()
