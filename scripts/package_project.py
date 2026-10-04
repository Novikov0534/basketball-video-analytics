"""Собирает лёгкий архив исходников без окружений, пользовательских данных и весов."""

from pathlib import Path
import os
import zipfile

root = Path(__file__).resolve().parents[1]
target = root.parent / "basketball-cv.zip"
skip_dirs = {
    ".git", ".venv", ".venv-rf", ".sam2", "__pycache__", ".pytest_cache",
    ".ruff_cache", "outputs", "uploads", "drive", ".gradio",
}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.endswith(".egg-info")]
        for name in sorted(files):
            p = Path(directory) / name
            if name == ".env" or p.suffix in {".pt", ".onnx", ".engine", ".pyc", ".log", ".zip"}:
                continue
            archive.write(p, Path(root.name) / p.relative_to(root))
print(target)
print(f"{target.stat().st_size / 1024 / 1024:.2f} MiB")
