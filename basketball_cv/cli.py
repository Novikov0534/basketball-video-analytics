"""Командная строка: basketball-cv ui | analyze | demo | doctor."""

import argparse
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def demo_setup():
    """Конфигурация и детектор синтетического демо (проверка программы)."""
    from .config import Config
    from .perception import CachedPerception

    cfg = Config.load(ROOT / "examples/demo_config.json")
    return ROOT / "examples/demo.mp4", cfg, CachedPerception(ROOT / "examples/demo_perception.jsonl")


def main():
    # Colab передаёт inline-бэкенд matplotlib, которого нет в нашем окружении
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(description="Basketball CV")
    commands = parser.add_subparsers(dest="command", required=True)
    ui = commands.add_parser("ui", help="веб-интерфейс")
    ui.add_argument("--share", action="store_true")
    ui.add_argument("--port", type=int, default=7860)
    analyze = commands.add_parser("analyze", help="обработать видео")
    analyze.add_argument("video")
    analyze.add_argument("--config")
    analyze.add_argument("--output", default="outputs")
    analyze.add_argument("--seconds", type=float)
    analyze.add_argument("--backend", choices=["roboflow", "roboflow_api", "yolo"])
    demo = commands.add_parser("demo", help="синтетическая проверка программы")
    demo.add_argument("--output", default="outputs")
    commands.add_parser("doctor", help="проверка окружения")
    args = parser.parse_args()

    if args.command == "ui":
        os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
        from .app import OUTPUTS, build_app

        auth = None
        if args.share:
            password = os.getenv("BASKETBALL_APP_PASSWORD") or secrets.token_urlsafe(12)
            print(f"Логин: analyst\nПароль: {password}", flush=True)
            auth = ("analyst", password)
        build_app().launch(server_name="127.0.0.1", server_port=args.port, share=args.share, auth=auth,
                           allowed_paths=[str(OUTPUTS), str(ROOT / "examples")], show_error=True)
        return

    if args.command == "doctor":
        import shutil
        import sys

        from .ocr import tesseract_command
        from .perception import worker_python

        print("Python:", sys.version.split()[0])
        try:
            import torch

            print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.is_available())
        except ImportError:
            print("PyTorch: не установлен")
        for package in ("supervision", "ultralytics", "gradio", "transformers"):
            try:
                module = __import__(package)
                print(f"{package}:", getattr(module, "__version__", "?"))
            except ImportError:
                print(f"{package}: не установлен")
        rf = worker_python()
        print("Окружение Roboflow:", rf if rf.exists() else "не установлено (bootstrap.py --roboflow)")
        print("SAM2:", describe_sam2())
        jersey = ROOT / "models" / "jersey_resnet.pt"
        print("ResNet номеров:", jersey if jersey.is_file() else "веса не найдены (будет использован SmolVLM2)")
        print("Ключ ROBOFLOW_API_KEY:", "задан" if os.getenv("ROBOFLOW_API_KEY") else "нет")
        print("FFmpeg:", shutil.which("ffmpeg"), "· Tesseract:", tesseract_command())
        return

    from .config import Config
    from .pipeline import analyze as run

    detector = None
    if args.command == "demo":
        video, cfg, detector = demo_setup()
    else:
        cfg = Config.load(args.config) if args.config else Config()
        if args.seconds is not None:
            cfg.max_seconds = args.seconds
        if args.backend:
            cfg.backend = args.backend
        video = args.video
    folder, summary = run(video, cfg, args.output, detector=detector,
                          progress=lambda fraction, message, frame=None: print(f"{fraction:6.0%} {message}", flush=True))
    score = summary["score"]
    print(f"\nСчёт: {summary['team_names'][0]} {score[0]} : {score[1]} {summary['team_names'][1]}")
    print(f"Игроков: {summary['player_count']} · событий: {summary['event_count']}")
    print("Результат:", folder)


def describe_sam2():
    """Строка о доступности SAM2 для команды doctor."""
    import subprocess

    from .perception import sam2_files, worker_python

    checkpoint, _ = sam2_files()
    if checkpoint is None:
        return "не установлен (bootstrap.py --roboflow --sam2); отслеживание выполнит ByteTrack"
    python = worker_python()
    if not python.exists():
        return f"веса есть ({checkpoint.name}), но нет окружения .venv-rf"
    probe = ("import sys; sys.path.insert(0, %r); import sam2.build_sam as b;"
             "print('поток' if hasattr(b, 'build_sam2_camera_predictor') else 'видеорежим')"
             % str(checkpoint.parent.parent))
    result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True,
                            env=dict(os.environ, MPLBACKEND="Agg"))
    if result.returncode:
        return f"веса есть ({checkpoint.name}), но модуль не импортируется"
    return f"{checkpoint.name}, режим: {result.stdout.strip()}"


if __name__ == "__main__":
    main()
