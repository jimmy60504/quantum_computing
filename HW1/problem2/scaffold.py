"""CLI scaffold for QCAA HW1 Problem 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import DATASET_LOADERS, METHOD_SPECS, ensure_output_dirs, make_result_skeleton


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QCAA HW1 Problem 2 scaffold")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("HW1") / "problem2",
        help="Directory to store scaffold outputs.",
    )
    parser.add_argument(
        "--preview-datasets",
        action="store_true",
        help="Generate dataset summaries for circle and moons.",
    )
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="Write a JSON scaffold plan for the three methods and two datasets.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paths = ensure_output_dirs(args.output_dir)

    print("QCAA HW1 Problem 2 scaffold")
    print(f"output_dir={paths['root']}")
    print("methods:")
    for spec in METHOD_SPECS.values():
        print(f"  - {spec.id}: {spec.label}")
        print(f"    {spec.summary}")

    print("datasets:")
    for dataset_name in DATASET_LOADERS:
        print(f"  - {dataset_name}")

    if args.preview_datasets:
        from .benchmark import summarize_dataset

        summaries = {}
        for dataset_name, loader in DATASET_LOADERS.items():
            bundle = loader()
            summary = summarize_dataset(bundle)
            summaries[dataset_name] = {
                "train_size": summary.train_size,
                "test_size": summary.test_size,
                "class_balance_train": summary.class_balance_train,
                "class_balance_test": summary.class_balance_test,
            }

        preview_path = paths["results"] / "dataset_preview.json"
        preview_path.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"wrote dataset preview: {preview_path}")

    if args.write_plan:
        plan = {
            "datasets": list(DATASET_LOADERS.keys()),
            "methods": [spec.__dict__ for spec in METHOD_SPECS.values()],
            "result_rows": [
                make_result_skeleton(dataset_name, method_id).to_dict()
                for dataset_name in DATASET_LOADERS
                for method_id in METHOD_SPECS
            ],
            "required_outputs": {
                "decision_boundary_grid": "3 x 2 plots",
                "comparison_table": [
                    "test_accuracy",
                    "trainable_parameters_or_kernel_evaluations",
                    "training_time_seconds",
                ],
            },
        }
        plan_path = paths["results"] / "benchmark_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"wrote benchmark plan: {plan_path}")


if __name__ == "__main__":
    main()
