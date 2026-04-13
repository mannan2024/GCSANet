"""CLI entry point for the GCSANet experiment pipeline."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
STAGE_SCRIPTS = {
    "setup": "01_setup_data.py",
    "model": "02_model_losses.py",
    "train": "03_training_eval.py",
    "ablation": "04_ablation_studies.py",
}


def run_stage(stage: str, namespace: dict | None = None) -> dict:
    script_path = BASE_DIR / STAGE_SCRIPTS[stage]
    return runpy.run_path(str(script_path), init_globals=namespace or {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the GCSANet pipeline from data setup through ablation studies."
    )
    parser.add_argument(
        "--stage",
        choices=["setup", "model", "train", "ablation", "all"],
        default="all",
        help="Pipeline stage to execute.",
    )
    args = parser.parse_args()

    if args.stage == "all":
        shared_namespace: dict = {}
        for stage in ("setup", "model", "train", "ablation"):
            shared_namespace.update(run_stage(stage, shared_namespace))
        return

    run_stage(args.stage)


if __name__ == "__main__":
    main()
