"""Reproducible project environment; never alter Colab's torch/Pillow installation."""

from pathlib import Path
import argparse
import os
import shutil
import urllib.request
import hashlib
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]

# Our trained jersey-number model is stored as a GitHub Release asset.
# The project ZIP remains small; Colab downloads the model server-to-server.
DEFAULT_JERSEY_RESNET_URL = (
    "https://github.com/Novikov0534/basketball-video-analytics/"
    "releases/download/models/jersey_resnet.pt"
)
DEFAULT_JERSEY_RESNET_SHA256 = "c7af17c90e730cc75bc2cd71c078c13dded8445fdb7d165999af29c1164e4b86"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--roboflow",
        action="store_true",
        help="дополнительно создать .venv-rf с пакетом inference для моделей Roboflow",
    )
    parser.add_argument(
        "--sam2",
        action="store_true",
        help="дополнительно поставить SAM2 в .venv-rf для устойчивого отслеживания игроков",
    )
    parser.add_argument(
        "--jersey-url",
        default="",
        help="URL готовых весов jersey_resnet.pt; также можно задать JERSEY_RESNET_URL",
    )
    args = parser.parse_args()
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise SystemExit(
            "Нужен Python 3.11 или 3.12. В Colab используйте готовый ноутбук проекта."
        )
    folder = ROOT / ".venv"
    python = folder / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(folder)
    run = lambda *args: subprocess.run([str(python), *args], cwd=ROOT, check=True)
    run("-m", "pip", "install", "pip==25.2", "setuptools==80.9.0", "wheel==0.45.1")
    index = "https://download.pytorch.org/whl/" + (
        "cu126" if args.device == "cuda" else "cpu"
    )
    run(
        "-m",
        "pip",
        "install",
        "torch==2.8.0",
        "torchvision==0.23.0",
        "-c",
        "constraints.txt",
        "--index-url",
        index,
    )
    run("-m", "pip", "install", "-r", "requirements.txt", "-c", "constraints.txt")
    run("-m", "pip", "install", "--no-deps", "-e", ".")
    run("-m", "pip", "check")
    if args.roboflow or args.sam2:
        install_roboflow_environment(args.device)
    if args.sam2:
        try:
            install_sam2()
        except Exception as exc:  # SAM2 необязателен: без него работает ByteTrack
            print(f"\nSAM2 установить не удалось: {exc}\n"
                  "Это не критично — отслеживание выполнит ByteTrack, остальное работает.\n",
                  flush=True)
    ensure_jersey_weights(args.jersey_url)
    run("-m", "basketball_cv.cli", "doctor")
    print(
        "\nГотово. Запуск интерфейса:\n", python, "-m basketball_cv.cli ui", flush=True
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_jersey_weights(url=""):
    """Connect pre-trained jersey ResNet weights without retraining.

    Priority: existing ``models/jersey_resnet.pt`` → ``JERSEY_RESNET_PATH``
    → URL from ``--jersey-url``/``JERSEY_RESNET_URL`` → the project's public
    GitHub Release. The built-in asset checksum is verified.
    """
    target = ROOT / "models" / "jersey_resnet.pt"
    target.parent.mkdir(exist_ok=True)
    if target.is_file() and target.stat().st_size > 1_000_000:
        print("✓ ResNet для номеров уже есть:", target, flush=True)
        return target

    local = os.getenv("JERSEY_RESNET_PATH", "").strip()
    if local:
        source = Path(local).expanduser()
        if source.is_file():
            shutil.copy2(source, target)
            print("✓ Скопированы готовые веса ResNet:", target, flush=True)
            return target
        print("JERSEY_RESNET_PATH не найден:", source, flush=True)

    explicit_url = (url or os.getenv("JERSEY_RESNET_URL", "")).strip()
    download_url = explicit_url or DEFAULT_JERSEY_RESNET_URL
    enforce_checksum = not explicit_url
    temporary = target.with_suffix(".pt.part")

    def progress(blocks, block_size, total):
        if total <= 0:
            return
        done = min(total, blocks * block_size)
        percent = int(done * 100 / total)
        if percent >= progress.last + 10 or done == total:
            print(f"  ResNet: {percent:3d}% ({done / 1024**2:.1f}/{total / 1024**2:.1f} MB)", flush=True)
            progress.last = percent
    progress.last = -10

    try:
        print("↓ Скачиваю наши веса jersey_resnet.pt из GitHub Release...", flush=True)
        urllib.request.urlretrieve(download_url, temporary, reporthook=progress)
        if temporary.stat().st_size < 50_000_000:
            raise RuntimeError("скачанный файл слишком мал и не похож на обученные веса")
        if enforce_checksum:
            actual = _sha256(temporary)
            if actual != DEFAULT_JERSEY_RESNET_SHA256:
                raise RuntimeError("SHA-256 скачанных весов не совпал с опубликованной моделью")
        temporary.replace(target)
        print(f"✓ ResNet готов: {target} ({target.stat().st_size / 1024**2:.1f} MB)", flush=True)
        return target
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        print(f"Не удалось скачать jersey_resnet.pt: {exc}. Используется fallback SmolVLM2.", flush=True)
        return None



SAM2_REPOSITORY = "https://github.com/Gy920/segment-anything-2-real-time.git"
SAM2_WEIGHTS = (
    "sam2.1_hiera_small.pt",
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
)


def install_sam2():
    """SAM2 для отслеживания игроков (необязательный компонент).

    Ставится в то же окружение .venv-rf, что и модели Roboflow: обеим нужен
    один torch. Две тонкости, на которых спотыкается установка из оригинального
    ноутбука:

    * --no-build-isolation — setup.py пакета импортирует torch при сборке, а в
      изолированном окружении pip его не видит;
    * SAM2_BUILD_CUDA=0 — отключает компиляцию CUDA-расширения. Оно отвечает
      только за постобработку масок (удаление мелких дыр), которую мы делаем
      сами через OpenCV, зато его сборка требует совпадения версий CUDA и
      компилятора и падает чаще всего остального.

    Если установка пакета всё-таки не удалась, SAM2 подключается напрямую из
    папки с исходниками: код на Python полностью рабочий, собирать нечего.
    """
    folder = ROOT / ".sam2"
    python = ROOT / ".venv-rf" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    marker = folder / "sam2" / "build_sam.py"
    if folder.exists() and not marker.is_file():
        # неполный клон с прошлой неудачной попытки: начинаем заново
        print("Папка .sam2 повреждена, клонирую заново", flush=True)
        shutil.rmtree(folder, ignore_errors=True)
    if not folder.exists():
        subprocess.run(["git", "clone", "--depth", "1", SAM2_REPOSITORY, str(folder)], check=True)
    if "build_sam2_camera_predictor" not in marker.read_text(encoding="utf-8", errors="ignore"):
        raise SystemExit("В склонированном SAM2 нет потокового интерфейса — проверьте адрес репозитория.")
    environment = dict(os.environ, MPLBACKEND="Agg", SAM2_BUILD_CUDA="0", SAM2_BUILD_ALLOW_ERRORS="1")

    def attempt(*arguments):
        result = subprocess.run([str(python), *arguments], cwd=folder, env=environment,
                                capture_output=True, text=True)
        if result.returncode:
            tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:])
            print(f"  не удалось ({' '.join(arguments[-3:])}):\n{tail}", flush=True)
        return result.returncode == 0

    subprocess.run([str(python), "-m", "pip", "install", "ninja"], cwd=folder, env=environment, check=False)
    installed = (attempt("-m", "pip", "install", "-e", ".", "--no-build-isolation")
                 or attempt("-m", "pip", "install", ".", "--no-build-isolation"))
    if not installed:
        # запасной путь: ставим только зависимости, сам пакет берём из исходников
        print("Ставлю SAM2 из исходников без сборки пакета", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "hydra-core", "iopath", "pillow", "tqdm"],
                       cwd=folder, env=environment, check=True)
    weights = download_sam2_weights(folder)
    probe = (
        "import sys; sys.path.insert(0, '.');"
        "import sam2.build_sam as b;"
        "print('SAM2 OK', b.__file__);"
        "print('потоковый интерфейс:', hasattr(b, 'build_sam2_camera_predictor'))"
    )
    check = subprocess.run([str(python), "-c", probe], cwd=folder, env=environment,
                           capture_output=True, text=True)
    if check.returncode:
        raise SystemExit("SAM2 не импортируется:\n" + (check.stdout + check.stderr)[-1500:])
    print(check.stdout.strip(), flush=True)
    if "потоковый интерфейс: False" in check.stdout:
        print("Внимание: доступен только официальный SAM2 без потокового интерфейса. "
              "Отслеживание будет работать в видеорежиме (медленнее, но результат тот же).", flush=True)
    print("SAM2 готов:", weights, flush=True)


def download_sam2_weights(folder):
    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    name, url = SAM2_WEIGHTS
    weights = checkpoints / name
    if not weights.is_file():
        print(f"Скачиваю веса SAM2: {name}", flush=True)
        urllib.request.urlretrieve(url, weights)
    return weights


def install_roboflow_environment(device):
    """Отдельное окружение для моделей Roboflow.

    Пакет inference тянет собственные torch, supervision и pillow, которые
    несовместимы с закреплёнными версиями приложения, поэтому он ставится в
    изолированное окружение .venv-rf. Оттуда запускается только
    basketball_cv/perception_worker.py, а результат передаётся файлом.
    """
    folder = ROOT / ".venv-rf"
    python = folder / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(folder)
    package = "inference-gpu==1.7.1" if device == "cuda" else "inference==1.7.1"
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], cwd=ROOT, check=True)
    subprocess.run([str(python), "-m", "pip", "install", package], cwd=ROOT, check=True)
    subprocess.run([str(python), "-c", "import inference; print('inference OK')"], cwd=ROOT, check=True)
    print("Окружение Roboflow готово:", python, flush=True)


if __name__ == "__main__":
    main()
