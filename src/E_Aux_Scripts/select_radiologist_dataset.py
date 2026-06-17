import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select 70 counterfactual and 30 healthy mammograms"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/images/VinDrMammo-CLIP-512-CLAHE",
        help="Root directory for original images",
    )
    parser.add_argument(
        "--cf_dir",
        type=str,
        default="data/images/repaint_results",
        help="Directory for counterfactual images",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/metadata/processed_df_birads_512.json",
        help="JSON file with metadata",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/images/radiologist/radiologist_assessment_dataset",
        help="Output directory for selected images",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/metadata/radiologist_dataset.json",
        help="Output JSON file for selected images metadata",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    return parser.parse_args()


def find_image(image_dir: Path, image_id: str) -> Path:
    """
    Finds an image in image_dir. This supports either:
    - image_dir/image_id
    - recursive search under image_dir
    """
    direct_path = image_dir / image_id
    if direct_path.exists():
        return direct_path

    matches = list(image_dir.rglob(image_id))
    if len(matches) == 0:
        raise FileNotFoundError(f"Could not find {image_id} under {image_dir}")

    if len(matches) > 1:
        print(f"Warning: multiple matches found for {image_id}. Using {matches[0]}")

    return matches[0]


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    cf_dir = Path(args.cf_dir)
    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)
    output_json = Path(args.output_json)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    cf_entries = [entry for entry in metadata if entry.get("has_cf") == 1]
    healthy_entries = [entry for entry in metadata if entry.get("healthy") == 1]

    if len(cf_entries) < 70:
        raise ValueError(f"Need 70 counterfactual entries, but found only {len(cf_entries)}")

    if len(healthy_entries) < 30:
        raise ValueError(f"Need 30 healthy entries, but found only {len(healthy_entries)}")

    selected_cf = random.sample(cf_entries, 70)
    selected_healthy = random.sample(healthy_entries, 30)

    selected_metadata = []

    for entry in selected_cf:
        image_id = entry["image_id"]

        src_path = find_image(cf_dir, image_id)
        dst_path = output_dir / image_id

        shutil.copy2(src_path, dst_path)

        new_entry = dict(entry)
        new_entry["source"] = "counterfactual"
        new_entry["copied_from"] = str(src_path)
        new_entry["copied_to"] = str(dst_path)

        selected_metadata.append(new_entry)

    for entry in selected_healthy:
        image_id = entry["image_id"]

        src_path = find_image(data_dir, image_id)
        dst_path = output_dir / image_id

        shutil.copy2(src_path, dst_path)

        new_entry = dict(entry)
        new_entry["source"] = "healthy_original"
        new_entry["copied_from"] = str(src_path)
        new_entry["copied_to"] = str(dst_path)

        selected_metadata.append(new_entry)

    with open(output_json, "w") as f:
        json.dump(selected_metadata, f, indent=2)

    print(f"Copied {len(selected_cf)} counterfactual images from {cf_dir}")
    print(f"Copied {len(selected_healthy)} healthy images from {data_dir}")
    print(f"Saved {len(selected_metadata)} metadata entries to {output_json}")
    print(f"Images copied to {output_dir}")


if __name__ == "__main__":
    main()