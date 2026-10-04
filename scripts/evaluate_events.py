import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basketball_cv.report import read_csv
from basketball_cv.evaluation import evaluate

parser = argparse.ArgumentParser(
    description="Сопоставить события с ручной разметкой CSV"
)
parser.add_argument("predicted")
parser.add_argument("reference")
parser.add_argument("--tolerance", type=float, default=1.0)
parser.add_argument("--match-player", action="store_true")
args = parser.parse_args()
print(
    json.dumps(
        evaluate(
            read_csv(args.predicted),
            read_csv(args.reference),
            args.tolerance,
            args.match_player,
        ),
        ensure_ascii=False,
        indent=2,
    )
)
