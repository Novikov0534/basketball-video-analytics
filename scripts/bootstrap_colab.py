"""Fast Google Colab setup for Basketball CV.

The Colab runtime already ships with Python, CUDA-enabled PyTorch, NumPy,
OpenCV and many scientific packages. Reusing them avoids downloading another
Python runtime and another full PyTorch stack on every fresh VM.

Roboflow Inference remains in its own environment because it has a much
heavier and stricter dependency stack than the main app.  On Colab Python
3.11/3.12 that worker environment reuses Colab's already installed PyTorch.
On newer Colab images (for example Python 3.13) the main application still
uses the fast system environment, while only the Roboflow worker gets a small
Python 3.12 runtime installed with uv.  This avoids rebuilding the whole app
environment just because Roboflow does not support the newest Python yet.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parents[1]
RF_ENV = ROOT / ".venv-rf"
SAM2_DIR = ROOT / ".sam2"

# In Colab we deliberately use minimum versions instead of old upper pins.
# The runtime already contains a mutually compatible set (for example Gradio 6,
# Transformers 5 and huggingface-hub 1.x on Python 3.13). Downgrading those
# packages is both slow and can break other preinstalled Colab components.
APP_REQUIREMENTS = (
    ("ultralytics", "ultralytics", ">=8.3.228"),
    ("supervision", "supervision", "==0.27.0"),
    ("gradio", "gradio", ">=5.49.1"),
    ("huggingface-hub", "huggingface_hub", ">=0.36"),
    ("transformers", "transformers", ">=4.57.1"),
    ("pandas", "pandas", ">=2.2"),
    ("pytesseract", "pytesseract", ">=0.3.13"),
    ("requests", "requests", ">=2.32"),
    ("umap-learn", "umap", ">=0.5.9"),
)

BASE_IMPORTS = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("opencv-python", "cv2"),
    ("pillow", "PIL"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
)


def run(command, *, cwd=ROOT, env=None, check=True):
    """Run a command while streaming output to the notebook."""
    merged = dict(os.environ, MPLBACKEND="Agg", PIP_DISABLE_PIP_VERSION_CHECK="1")
    if env:
        merged.update(env)
    print("\n$", " ".join(map(str, command)), flush=True)
    return subprocess.run([str(x) for x in command], cwd=cwd, env=merged, check=check)


def import_ok(module):
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def requirement_satisfied(distribution, module, specifier):
    if not import_ok(module):
        return False
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        version = importlib.metadata.version(distribution)
        return Version(version) in SpecifierSet(specifier)
    except Exception:
        # If metadata is unusual but the module imports, leave it alone. The
        # final doctor/probe will catch a real incompatibility.
        return True


def ensure_system_tools():
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if shutil.which("tesseract") is None:
        missing.append("tesseract-ocr")
    if not missing:
        print("✓ FFmpeg и Tesseract уже есть в Colab", flush=True)
        return
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", *missing])


def ensure_base_runtime(device):
    missing = [dist for dist, module in BASE_IMPORTS if not import_ok(module)]
    # In normal Colab these are already present. Only repair an unusual image.
    repairs = []
    if "numpy" in missing:
        repairs.append("numpy>=2,<3")
    if "scipy" in missing:
        repairs.append("scipy>=1.10,<2")
    if "opencv-python" in missing:
        repairs.append("opencv-python-headless>=4.8,<5")
    if "pillow" in missing:
        repairs.append("pillow>=10,<13")
    if repairs:
        run([sys.executable, "-m", "pip", "install", "--prefer-binary", *repairs])

    if not import_ok("torch") or not import_ok("torchvision"):
        raise RuntimeError(
            "В этой среде Colab нет готового PyTorch/torchvision. "
            "Выберите стандартную GPU-среду Colab и перезапустите ячейку."
        )
    import torch
    print(f"✓ Используется готовый PyTorch Colab: {torch.__version__}", flush=True)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна. В Colab выберите GPU в настройках среды выполнения.")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}", flush=True)


def ensure_app_packages():
    missing = []
    for distribution, module, specifier in APP_REQUIREMENTS:
        if requirement_satisfied(distribution, module, specifier):
            try:
                version = importlib.metadata.version(distribution)
            except Exception:
                version = "installed"
            print(f"✓ {distribution} {version}", flush=True)
        else:
            missing.append(f"{distribution}{specifier}")

    # Socks support is an optional httpx extra and is tiny. Install it only if absent.
    if not import_ok("socksio"):
        missing.append("socksio>=1.0,<2")

    if missing:
        print("Устанавливаю только недостающие/несовместимые пакеты приложения:", flush=True)
        print("  " + "\n  ".join(missing), flush=True)
        run([
            sys.executable, "-m", "pip", "install", "--prefer-binary",
            "--upgrade-strategy", "only-if-needed", *missing,
        ])
    else:
        print("✓ Пакеты приложения уже подходят — pip install пропущен", flush=True)

    # No editable install is needed in Colab. The notebook always runs commands
    # from the project root, so ``python -m basketball_cv...`` imports the local
    # package directly. Skipping pip -e also avoids build isolation and Python
    # metadata checks on a brand-new Colab runtime.
    print("✓ Код проекта используется прямо из /content/basketball-cv", flush=True)


def rf_python():
    return RF_ENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _python_minor(python):
    result = subprocess.run(
        [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return ""
    return result.stdout.strip()


def inference_probe(python):
    result = subprocess.run(
        [str(python), "-c", "import inference; print('inference', getattr(inference, '__version__', 'OK'))"],
        cwd=ROOT,
        env=dict(os.environ, MPLBACKEND="Agg"),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("✓", result.stdout.strip(), flush=True)
        return True
    return False


def ensure_uv():
    """Return the uv executable, installing its wheel only when needed."""
    executable = shutil.which("uv")
    if executable:
        print("✓ uv уже доступен", flush=True)
        return executable
    print("Устанавливаю быстрый менеджер пакетов uv...", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "uv>=0.8,<1"])
    executable = shutil.which("uv")
    if not executable:
        candidate = Path(sys.executable).resolve().parent / "uv"
        if candidate.exists():
            executable = str(candidate)
    if not executable:
        raise RuntimeError("uv установился, но исполняемый файл не найден.")
    return executable


def rf_env_can_reuse_colab():
    """Roboflow 1.7.1 is kept on Python 3.11/3.12.

    system-site-packages can only be shared safely when the worker has the same
    Python minor as the Colab kernel.  On Python 3.13+ we create only the worker
    on 3.12 instead of recreating the whole application environment.
    """
    return sys.version_info[:2] in ((3, 11), (3, 12))


def rf_env_matches_mode(reuse_colab):
    cfg = RF_ENV / "pyvenv.cfg"
    python = rf_python()
    if not cfg.is_file() or not python.exists():
        return False
    text = cfg.read_text(encoding="utf-8", errors="ignore").lower()
    has_system = "include-system-site-packages = true" in text
    if reuse_colab:
        return has_system and _python_minor(python) == f"{sys.version_info.major}.{sys.version_info.minor}"
    return (not has_system) and _python_minor(python) == "3.12"


def create_rf_env():
    reuse_colab = rf_env_can_reuse_colab()
    if RF_ENV.exists() and not rf_env_matches_mode(reuse_colab):
        print("Конфигурация .venv-rf не подходит текущему Colab — пересоздаю.", flush=True)
        shutil.rmtree(RF_ENV, ignore_errors=True)

    python = rf_python()
    if python.exists():
        return python, reuse_colab

    if reuse_colab:
        print("Создаю лёгкое окружение Roboflow с доступом к пакетам Colab...", flush=True)
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(RF_ENV)
        return rf_python(), True

    # Current Colab may move to Python 3.13 before inference-gpu does.  Do not
    # downgrade/rebuild the main app: install Python 3.12 only for its worker.
    print(
        f"Python Colab {sys.version_info.major}.{sys.version_info.minor}: "
        "основное приложение остаётся на нём; для Roboflow готовлю только Python 3.12.",
        flush=True,
    )
    uv = ensure_uv()
    run([uv, "python", "install", "3.12"])
    run([uv, "venv", "--python", "3.12", "--seed", str(RF_ENV)])
    python = rf_python()
    if not python.exists() or _python_minor(python) != "3.12":
        raise RuntimeError("Не удалось подготовить Python 3.12 для Roboflow.")
    return python, False


def torch_constraints():
    """Pin the worker to Colab torch only when both use the same Python minor."""
    import torch
    import torchvision
    torch_version = torch.__version__.split("+")[0]
    vision_version = torchvision.__version__.split("+")[0]
    path = ROOT / ".colab-torch-constraints.txt"
    path.write_text(
        f"torch=={torch_version}\ntorchvision=={vision_version}\n",
        encoding="utf-8",
    )
    return path


def _install_with_uv(python, packages, *, constraints=None):
    uv = ensure_uv()
    command = [uv, "pip", "install", "--python", str(python)]
    if constraints is not None:
        command += ["-c", str(constraints)]
    command += list(packages)
    run(command)


def install_roboflow_fast(device):
    python, reuse_colab = create_rf_env()
    if inference_probe(python):
        print("✓ Roboflow Inference уже установлен", flush=True)
        return python

    package = "inference-gpu==1.7.1" if device == "cuda" else "inference==1.7.1"

    if reuse_colab:
        constraints = torch_constraints()
        print("Устанавливаю Roboflow, используя готовый PyTorch Colab...", flush=True)
        try:
            _install_with_uv(python, [package], constraints=constraints)
        except subprocess.CalledProcessError:
            # The exact Colab torch may be newer than inference-gpu accepts.
            # Keep the main app untouched and fall back to a Python 3.12 worker.
            print(
                "Текущий PyTorch Colab не подходит Roboflow. "
                "Перехожу на отдельный Python 3.12 только для worker.",
                flush=True,
            )
            shutil.rmtree(RF_ENV, ignore_errors=True)
            uv = ensure_uv()
            run([uv, "python", "install", "3.12"])
            run([uv, "venv", "--python", "3.12", "--seed", str(RF_ENV)])
            python = rf_python()
            _install_with_uv(python, [package])
        finally:
            constraints.unlink(missing_ok=True)
    else:
        print("Устанавливаю Roboflow в Python 3.12 worker через uv...", flush=True)
        _install_with_uv(python, [package])

    if not inference_probe(python):
        raise RuntimeError("Roboflow Inference установился, но не импортируется.")
    return python


def install_sam2_source_only(python):
    """Install SAM2 without CUDA extension/package build.

    perception_worker.py imports the fork directly from .sam2, so compiling an
    editable package is unnecessary. This removes one of the slowest and most
    fragile setup steps.
    """
    marker = SAM2_DIR / "sam2" / "build_sam.py"
    if SAM2_DIR.exists() and not marker.is_file():
        shutil.rmtree(SAM2_DIR, ignore_errors=True)
    if not SAM2_DIR.exists():
        run([
            "git", "clone", "--depth", "1",
            "https://github.com/Gy920/segment-anything-2-real-time.git",
            SAM2_DIR,
        ])
    if not marker.is_file():
        raise RuntimeError("SAM2 repository is incomplete: sam2/build_sam.py not found")

    # Only dependencies that are not normally supplied by Roboflow/Colab.
    probe = subprocess.run(
        [str(python), "-c", "import hydra, iopath, tqdm; print('deps OK')"],
        cwd=SAM2_DIR,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        run([python, "-m", "pip", "install", "--prefer-binary", "hydra-core", "iopath", "tqdm"], cwd=SAM2_DIR)
    else:
        print("✓ Зависимости SAM2 уже установлены", flush=True)

    from bootstrap import download_sam2_weights
    weights = download_sam2_weights(SAM2_DIR)

    test = (
        "import sys; sys.path.insert(0, %r); "
        "import sam2.build_sam as b; "
        "print('SAM2 OK:', b.__file__)" % str(SAM2_DIR)
    )
    result = subprocess.run(
        [str(python), "-c", test], cwd=SAM2_DIR,
        env=dict(os.environ, MPLBACKEND="Agg", SAM2_BUILD_CUDA="0", SAM2_BUILD_ALLOW_ERRORS="1"),
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError("SAM2 не импортируется:\n" + (result.stdout + result.stderr)[-1800:])
    print("✓", result.stdout.strip(), flush=True)
    print("✓ SAM2 weights:", weights, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--roboflow", action="store_true")
    parser.add_argument("--sam2", action="store_true")
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        raise SystemExit("Нужен Python 3.11 или новее.")

    started = time.monotonic()
    print("Basketball CV — быстрая подготовка Colab", flush=True)
    print("Python:", sys.version.split()[0], flush=True)

    ensure_system_tools()
    ensure_base_runtime(args.device)
    ensure_app_packages()

    if args.roboflow or args.sam2:
        python = install_roboflow_fast(args.device)
        os.environ["BCV_ROBOFLOW_PYTHON"] = str(python)
        if args.sam2:
            install_sam2_source_only(python)

    # Shared downloader uses the project's permanent GitHub Release by default.
    from bootstrap import ensure_jersey_weights
    ensure_jersey_weights()

    # Doctor is intentionally last: setup failure is caught before UI launch.
    run([sys.executable, "-m", "basketball_cv.cli", "doctor"])
    print(f"\n✓ Готово за {(time.monotonic() - started) / 60:.1f} мин.", flush=True)
    print("Следующая ячейка запускает интерфейс.", flush=True)


if __name__ == "__main__":
    main()
