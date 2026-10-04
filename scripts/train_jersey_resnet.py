"""Обучение классификатора номеров на майках (ResNet).

Данные: датасет Roboflow `basketball-jersey-numbers-ocr` — кропы номеров с
плей-офф НБА. Скрипт принимает его в двух видах:

  * папки по классам (обычный формат классификации):
        train/7/img001.jpg, train/23/img002.jpg, valid/..., test/...
  * JSONL (исходный формат для языковой модели): строки вида
        {"image": "crops/img001.jpg", "suffix": "7"}
    — скрипт сам разложит их по папкам.

Запуск в Colab:
    .venv/bin/python scripts/train_jersey_resnet.py --data /content/jersey \\
        --epochs 25 --output models/jersey_resnet.pt

Результат: веса модели, список классов и отчёт о точности в
models/jersey_resnet_report.json — цифры для главы с экспериментами.
"""

from collections import Counter
from pathlib import Path
import argparse
import json
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_cv.jersey import IMAGE_SIZE, build_model, prepare_crop  # noqa: E402

SPLITS = ("train", "valid", "test")


def collect_from_folders(root):
    """{раздел: [(путь, класс)]} из структуры train/КЛАСС/файл.jpg."""
    data = {split: [] for split in SPLITS}
    for split in SPLITS:
        folder = root / split
        if not folder.is_dir():
            continue
        for class_folder in sorted(folder.iterdir()):
            if class_folder.is_dir():
                for image in class_folder.glob("*.*"):
                    if image.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        data[split].append((image, class_folder.name.strip()))
    return data


def collect_from_jsonl(root):
    """{раздел: [(путь, класс)]} из JSONL-описаний.

    Датасет номеров у Roboflow имеет тип text-image-pairs и выгружается
    только в JSONL: строка описывает изображение и правильный ответ. Ключи
    в разных выгрузках называются по-разному, поэтому проверяем несколько
    вариантов, а файл ищем и рядом с описанием, и в соседних папках.
    """
    data = {split: [] for split in SPLITS}
    missing = 0
    for path in sorted(root.rglob("*.jsonl")):
        split = next((s for s in SPLITS if s in path.parts or s in path.stem), "train")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            name = (record.get("image") or record.get("file_name") or record.get("path")
                    or record.get("image_path") or record.get("file"))
            label = str(record.get("suffix") or record.get("label") or record.get("text")
                        or record.get("answer") or "").strip()
            if not name or not label.isdigit():
                continue
            image = Path(name) if Path(name).is_absolute() else path.parent / name
            if not image.is_file():  # изображения могут лежать в подпапке рядом
                found = next((c for c in (path.parent / "images" / Path(name).name,
                                          root / Path(name).name) if c.is_file()), None)
                image = found or image
            if image.is_file():
                data[split].append((image, label))
            else:
                missing += 1
    if missing:
        print(f"Не найдено файлов изображений: {missing} (проверьте структуру датасета)")
    return data


def split_if_needed(data, seed=42):
    """Если разделов нет, делим обучающую часть на train/valid/test как 70/15/15."""
    if data["valid"] or data["test"]:
        return data
    items = list(data["train"])
    random.Random(seed).shuffle(items)
    first, second = int(0.7 * len(items)), int(0.85 * len(items))
    return {"train": items[:first], "valid": items[first:second], "test": items[second:]}


def read_images(items, classes):
    """Читает файлы с диска один раз: [(изображение, класс)].

    Перечитывать 3.6 тысячи файлов на каждой эпохе — самая долгая часть
    обучения, поэтому изображения держим в памяти как есть (uint8, около
    100 МБ), а аугментацию применяем уже к ним.
    """
    import cv2

    index = {name: position for position, name in enumerate(classes)}
    cache = []
    for path, label in items:
        image = cv2.imread(str(path))
        if image is not None and label in index:
            cache.append((image, index[label]))
    return cache


def prepare_batch(cache, classes, augment=False, seed=0):
    """Готовит массивы X, y из прочитанных изображений."""
    import cv2

    rng = np.random.default_rng(seed)
    features, labels = [], []
    for source, label in cache:
        image = source
        if augment:
            if rng.random() < 0.5:  # игроки видны под разными углами и в движении
                angle = rng.uniform(-8, 8)
                matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1.0)
                image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            if rng.random() < 0.4:  # смазывание от быстрого движения
                image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.4, 1.2))
            if rng.random() < 0.5:  # разное освещение зала
                image = np.clip(image * rng.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        prepared = prepare_crop(image)
        if prepared is not None:
            features.append(prepared)
            labels.append(label)
    return (np.stack(features) if features else np.empty((0, 3, IMAGE_SIZE, IMAGE_SIZE), np.float32),
            np.array(labels))


def evaluate(model, torch, features, labels, device, batch_size=128):
    if not len(features):
        return None, Counter()
    model.eval()
    correct, mistakes = 0, Counter()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            predicted = model(batch).argmax(dim=1).cpu().numpy()
            reference = labels[start:start + batch_size]
            correct += int((predicted == reference).sum())
            for got, want in zip(predicted, reference):
                if got != want:
                    mistakes[(int(want), int(got))] += 1
    return correct / len(features), mistakes


def main():
    parser = argparse.ArgumentParser(description="Обучение ResNet для чтения номеров")
    parser.add_argument("--data", required=True, help="папка датасета")
    parser.add_argument("--output", default="models/jersey_resnet.pt")
    parser.add_argument("--architecture", default="resnet34")
    parser.add_argument("--epochs", type=int, default=40,
                        help="верхняя граница; обучение остановится раньше, если точность перестанет расти")
    parser.add_argument("--patience", type=int, default=8,
                        help="сколько эпох ждать улучшения перед остановкой (0 — не останавливаться)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    # веса ImageNet ускоряют сходимость в разы; --no-pretrained нужен там,
    # где нет доступа к сети или требуется обучение с нуля для сравнения
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="стартовать с весов ImageNet (по умолчанию да)")
    options = parser.parse_args()

    import torch
    from torch import nn
    from torchvision import models as torchvision_models

    root = Path(options.data)
    # структура датасета заранее не известна: в выгрузке Roboflow папки train/valid
    # есть, но классы описаны в JSONL, а не именами вложенных папок. Поэтому
    # пробуем оба способа и берём тот, что дал больше размеченных изображений.
    by_folders = collect_from_folders(root)
    by_jsonl = collect_from_jsonl(root)
    data = split_if_needed(by_folders if sum(map(len, by_folders.values()))
                           >= sum(map(len, by_jsonl.values())) else by_jsonl)
    total = sum(len(items) for items in data.values())
    if total < 50:
        listing = sorted({p.suffix.lower() for p in root.rglob("*") if p.suffix})
        raise SystemExit(
            f"В {root} найдено всего {total} подходящих изображений.\n"
            f"Типы файлов в папке: {listing or 'папка пуста'}\n"
            "Ожидается либо структура train/КЛАСС/файл.jpg, либо выгрузка JSONL "
            "(scripts/download_jersey_dataset.py качает её автоматически)."
        )
    classes = sorted({label for items in data.values() for _, label in items}, key=lambda v: (len(v), v))
    print(f"Изображений: {total}, классов: {len(classes)}")
    print("Разделы:", {split: len(items) for split, items in data.items()})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(options.architecture, len(classes))
    if options.pretrained:
        try:
            factory = getattr(torchvision_models, options.architecture)
            pretrained = factory(weights="DEFAULT")
            state = {k: v for k, v in pretrained.state_dict().items() if not k.startswith("fc.")}
            model.load_state_dict(state, strict=False)
            print("Стартую с весов ImageNet", flush=True)
        except Exception as exc:
            print(f"Веса ImageNet недоступны ({type(exc).__name__}); обучение с нуля — "
                  "потребуется больше эпох", flush=True)
    model.to(device)

    print("Читаю изображения в память…", flush=True)
    train_cache = read_images(data["train"], classes)
    x_valid, y_valid = prepare_batch(read_images(data["valid"], classes), classes)
    x_test, y_test = prepare_batch(read_images(data["test"], classes), classes)
    # веса классов: в реальных матчах номера встречаются очень неравномерно
    counts = Counter(label for _, label in data["train"])
    weights = torch.tensor([1.0 / max(1, counts.get(name, 0)) ** 0.5 for name in classes],
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights / weights.mean(), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options.epochs)

    best_accuracy, best_epoch, history, started = 0.0, 0, [], time.monotonic()
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, options.epochs + 1):
        x_train, y_train = prepare_batch(train_cache, classes, augment=True, seed=epoch)
        order = np.random.default_rng(epoch).permutation(len(x_train))
        model.train()
        loss_sum = 0.0
        for start in range(0, len(order), options.batch_size):
            batch = order[start:start + options.batch_size]
            inputs = torch.from_numpy(x_train[batch]).to(device)
            targets = torch.from_numpy(y_train[batch]).to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        schedule.step()
        accuracy, _ = evaluate(model, torch, x_valid, y_valid, device)
        history.append(dict(epoch=epoch, loss=round(loss_sum / max(1, len(order)), 4),
                            valid_accuracy=None if accuracy is None else round(accuracy, 4)))
        print(f"эпоха {epoch:3d}  потери {history[-1]['loss']:.4f}  точность на валидации "
              f"{history[-1]['valid_accuracy']}", flush=True)
        if accuracy is not None and accuracy >= best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save(dict(state_dict=model.state_dict(), classes=classes,
                            architecture=options.architecture), output)
        # ранняя остановка: точность не растёт patience эпох подряд
        if options.patience and epoch - best_epoch >= options.patience:
            print(f"Точность не улучшается {options.patience} эпох — останавливаюсь "
                  f"(лучшая была на эпохе {best_epoch})", flush=True)
            break

    if output.is_file():
        model.load_state_dict(torch.load(output, map_location=device, weights_only=False)["state_dict"])
    if not len(x_test):  # в некоторых выгрузках нет отложенной части
        print("Раздела test нет — итоговая точность считается на валидации "
              "(для диплома лучше выделить отдельную выборку).", flush=True)
        x_test, y_test = x_valid, y_valid
    test_accuracy, mistakes = evaluate(model, torch, x_test, y_test, device)
    report = dict(
        architecture=options.architecture, classes=len(classes), images=total,
        split={split: len(items) for split, items in data.items()},
        epochs_planned=options.epochs, epochs_run=len(history), best_epoch=best_epoch,
        valid_accuracy=round(best_accuracy, 4),
        test_accuracy=None if test_accuracy is None else round(test_accuracy, 4),
        minutes=round((time.monotonic() - started) / 60, 1),
        top_confusions=[dict(expected=classes[a], predicted=classes[b], count=n)
                        for (a, b), n in mistakes.most_common(10)],
        history=history,
    )
    report_path = output.with_name(output.stem + "_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nТочность на тесте: {report['test_accuracy']}")
    print("Веса:", output)
    print("Отчёт:", report_path)


if __name__ == "__main__":
    main()
