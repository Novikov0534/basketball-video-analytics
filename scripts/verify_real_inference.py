"""Optional online smoke test: real YOLO on the upstream photographic sample."""

from pathlib import Path
import json
import sys
from urllib.request import urlopen
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basketball_cv.config import Config
from basketball_cv.detectors import YoloDetector
from basketball_cv.pipeline import analyze

folder = ROOT / "outputs/real-smoke"
folder.mkdir(parents=True, exist_ok=True)
data = urlopen(
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg", timeout=60
).read()
image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError("Не удалось скачать тестовое изображение")
image = cv2.resize(image, (480, 640))
cfg = Config(
    device="cpu", camera_mode="fixed", max_seconds=2, target_fps=5, image_size=640
)
detector = YoloDetector(cfg)
boxes = detector.detect(image)
assert any(d.kind == "player" for d in boxes), "Тестовое фото должно содержать людей"
path = folder / "photographic-sample.mp4"
writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 640))
for i in range(20):
    writer.write(image)
writer.release()
output, summary = analyze(path, cfg, folder, detector=detector)
assert summary["processed_frames"] == 10 and summary["track_count"] >= 1
(folder / "smoke_test.json").write_text(
    json.dumps(
        {
            "success": True,
            "device": "cpu",
            "photographic_person_detections": sum(d.kind == "player" for d in boxes),
            "processed_frames": summary["processed_frames"],
            "track_count": summary["track_count"],
            "note": "Installation/integration test only; not basketball accuracy evaluation",
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("YOLO smoke test passed:", output)
