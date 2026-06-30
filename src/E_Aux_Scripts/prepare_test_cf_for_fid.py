"""Prepare counterfactual and healthy image directories for FID evaluation."""

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data/metadata/processed_df_birads.json"
DEFAULT_CF_DIR = (
    PROJECT_ROOT / "data/images/repaint_results_912x1520"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/images/test_cf_for_fid"
DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "data/images/VinDrMammo-CLIP-512-CLAHE-912-1520"
)
DEFAULT_HEALTHY_OUTPUT_DIR = PROJECT_ROOT / "data/images/healthy_for_fid"
DEFAULT_TEST_HEALTHY_OUTPUT_DIR = PROJECT_ROOT / "data/images/test_healthy_for_fid"


def copy_test_counterfactuals(
    metadata_path: Path,
    cf_dir: Path,
    output_dir: Path,
) -> tuple[int, list[str]]:
    """Copy records with ``split == 'test'`` and ``has_cf == 1``."""
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        record
        for record in metadata
        if record.get("split") == "test" and record.get("has_cf") == 1
    ]

    copied = 0
    missing = []

    for record in selected:
        image_id = record.get("image_id")
        if not image_id:
            missing.append("<record without image_id>")
            continue

        source_path = cf_dir / image_id
        if not source_path.is_file():
            missing.append(image_id)
            continue

        shutil.copy2(source_path, output_dir / image_id)
        copied += 1

    return copied, missing


def copy_healthy_images(
    metadata_path: Path,
    data_dir: Path,
    healthy_output_dir: Path,
    test_healthy_output_dir: Path,
) -> tuple[int, int, list[str]]:
    """Copy all healthy images and the healthy test subset for FID evaluation."""
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    healthy_output_dir.mkdir(parents=True, exist_ok=True)
    test_healthy_output_dir.mkdir(parents=True, exist_ok=True)

    healthy_records = [record for record in metadata if record.get("healthy") == 1]
    copied_healthy = 0
    copied_test_healthy = 0
    missing = []

    for record in healthy_records:
        image_id = record.get("image_id")
        if not image_id:
            missing.append("<record without image_id>")
            continue

        source_path = data_dir / image_id
        if not source_path.is_file():
            missing.append(image_id)
            continue

        shutil.copy2(source_path, healthy_output_dir / image_id)
        copied_healthy += 1

        if record.get("split") == "test":
            shutil.copy2(source_path, test_healthy_output_dir / image_id)
            copied_test_healthy += 1

    return copied_healthy, copied_test_healthy, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy test counterfactuals and healthy images into directories "
            "for FID evaluation."
        )
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Metadata JSON path (default: {DEFAULT_METADATA_PATH}).",
    )
    parser.add_argument(
        "--cf_dir",
        type=Path,
        default=DEFAULT_CF_DIR,
        help=f"Counterfactual image directory (default: {DEFAULT_CF_DIR}).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Original image directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--healthy_output_dir",
        type=Path,
        default=DEFAULT_HEALTHY_OUTPUT_DIR,
        help=f"All-healthy destination (default: {DEFAULT_HEALTHY_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--test_healthy_output_dir",
        type=Path,
        default=DEFAULT_TEST_HEALTHY_OUTPUT_DIR,
        help=(
            "Test-healthy destination "
            f"(default: {DEFAULT_TEST_HEALTHY_OUTPUT_DIR})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copied, missing = copy_test_counterfactuals(
        metadata_path=args.metadata_path,
        cf_dir=args.cf_dir,
        output_dir=args.output_dir,
    )

    print(f"Copied {copied} test counterfactuals to {args.output_dir}")
    if missing:
        print(f"Skipped {len(missing)} missing images:")
        for image_id in missing:
            print(f"  {image_id}")

    copied_healthy, copied_test_healthy, missing_healthy = copy_healthy_images(
        metadata_path=args.metadata_path,
        data_dir=args.data_dir,
        healthy_output_dir=args.healthy_output_dir,
        test_healthy_output_dir=args.test_healthy_output_dir,
    )
    print(f"Copied {copied_healthy} healthy images to {args.healthy_output_dir}")
    print(
        f"Copied {copied_test_healthy} test healthy images to "
        f"{args.test_healthy_output_dir}"
    )
    if missing_healthy:
        print(f"Skipped {len(missing_healthy)} missing healthy images:")
        for image_id in missing_healthy:
            print(f"  {image_id}")


if __name__ == "__main__":
    main()
