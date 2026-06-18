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
        default="data/images/radiologist_assessment_dataset_test",
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

EXCLUDED_CF_IDS = {
    "2b76a18e45c8e62db919403c6c526ec4",
    "3f41e1acc2be249bac2dc601633f7e99",
    "4c3a8a7db202ea367baca51aa5b3b6d5",
    "15ad456e4bc707e7ffeee20e60a3a820",
    "17c9d5489c19c75ed3cb4032ba49afb7",
    "19c0ecd06984b4d4b2c46812e85a06e6",
    "63f4053da6b026f6b58b159fdf79b6ab",
    "29fa27705317bd108ed3b927fdc4e3a0",
    "64e88bc3716211bb2fc5dae3ca9e5b92",
    "65cc9e7db32c047a5b13ed81767ac9fb",
    "212bc03533346819d0c6d449d631f69f",
    "09025dbf5abd08828ecd4ebf03724341",
    "f56a57a109aed21840b83c8453e06a6e",
    "2b76a18e45c8e62db919403c6c526ec4",
    "4d96d394fe4b20778014bdde96b5d34e",
    "15ad456e4bc707e7ffeee20e60a3a820",
    "19c0ecd06984b4d4b2c46812e85a06e6",
    "264d5e05c5774d9770eaf74c0483bd50",
    "09025dbf5abd08828ecd4ebf03724341",
    "15ad456e4bc707e7ffeee20e60a3a820",
    "19c0ecd06984b4d4b2c46812e85a06e6",
    "264d5e05c5774d9770eaf74c0483bd50",
    "5303d45e31ed5f069964b4c47a0f1b55",
    "6587b6ecbcf196b298b5898f7ad4763b",
    "bc58e54ec75acfc3acfb248f409ff2f4",
    "87811fa5ebde0c8e76add8f3b58f0d1c",
    "aa02e8c5422114d20da11c573d0ee8ee",
    "1cbc0ae5d67abccd58a4ba6d657d921e",
    "f3f7753468997b03b3d8707edd9bef37",
    "3fde15ab69283b96c1c24538427e7212",
    "851af5bfe4d9b12c605998e1fe7355c3",
    "870ec0cc5a1b1874d8c972a8595f41a2",
    "dbe631b24f8759c7f7022513582c39ba"
}

def image_stem(image_id: str) -> str:
    return Path(image_id).stem

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

    cf_entries = [
        entry
        for entry in metadata
        if entry.get("has_cf") == 1
        and entry.get("split") == "test"
        and image_stem(entry["image_id"]) not in EXCLUDED_CF_IDS
    ]
    healthy_entries = [entry for entry in metadata if entry.get("healthy") == 1 and entry.get("split") == "test"]

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