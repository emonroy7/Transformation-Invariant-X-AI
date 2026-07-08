"""Unified CLI entry point for the thesis experiments.

Usage:
    python scripts/run_experiment.py exp1 --model resnet152 --image img.jpg --transform rotation
    python scripts/run_experiment.py exp2 --model densenet201 --dataset coco
    python scripts/run_experiment.py exp3 --model xception              # MS COCO, all settings
    python scripts/run_experiment.py exp4 --model resnet152             # CAMELYON16 (zoom-invariant)
"""
import sys
from pathlib import Path

# Make the project root importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"exp1", "exp2", "exp3", "exp4"}:
        print(__doc__)
        sys.exit(1)

    which = sys.argv.pop(1)          # remove the subcommand; leave the rest for argparse
    if which == "exp1":
        from experiments.exp1_single_image_consistency import main as run
    elif which == "exp2":
        from experiments.exp2_class_specific import main as run
    elif which == "exp3":
        from experiments.exp3_dataset_level import main as run
    else:
        from experiments.exp4_histopathology_evaluation import main as run
    run()


if __name__ == "__main__":
    main()
