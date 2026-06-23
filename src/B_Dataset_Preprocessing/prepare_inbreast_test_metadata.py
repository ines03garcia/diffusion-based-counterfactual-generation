import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare INbreast metadata for classifier testing."
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        required=True,
        help="Path to INbreast metadata file (.csv, .xls, or .xlsx).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing INbreast images.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output CSV path for test metadata.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively search for images in subdirectories.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail if an image cannot be found or opened.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split value for all generated rows (default: test).",
    )
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    suffix = metadata_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(metadata_path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(metadata_path)
    raise ValueError(
        f"Unsupported metadata extension '{suffix}'. Use .csv, .xls, or .xlsx."
    )


def normalize_birads(value) -> Tuple[str, int]:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty Bi-Rads value")

    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse Bi-Rads value: '{value}'")

    birads_num = int(digits[0])
    return f"BI-RADS {birads_num}", birads_num


def build_image_index(images_dir: Path, recursive: bool) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}

    if recursive:
        candidates = [p for p in images_dir.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in images_dir.iterdir() if p.is_file()]

    for path in candidates:
        index.setdefault(path.name.lower(), []).append(path)

    return index


def resolve_image_path(
    file_name: str,
    images_dir: Path,
    image_index: Dict[str, List[Path]],
) -> Optional[Path]:
    name = str(file_name).strip()
    if not name:
        return None

    basename = Path(name).name
    candidates = image_index.get(basename.lower(), [])

    if not candidates and "." not in basename:
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidates = image_index.get(f"{basename}{ext}".lower(), [])
            if candidates:
                break

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Deterministic choice when duplicated names exist in different folders.
    return sorted(candidates)[0]


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()

    metadata_path = Path(args.metadata_path)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    df = load_metadata(metadata_path)

    required_columns = ["File Name", "Bi-Rads"]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required metadata columns: {missing_columns}. "
            "Expected at least 'File Name' and 'Bi-Rads'."
        )

    image_index = build_image_index(images_dir, recursive=args.recursive)

    output_rows = []
    missing_files = []
    unreadable_files = []
    invalid_birads = []

    df = df[df["File Name"].notnull()]
    # Convert File Name to int and then to string
    df["File Name"] = df["File Name"].apply(lambda x: str(int(x)))
    print(df["File Name"])

    for _, row in df.iterrows():
        file_name = row["File Name"]
        birads_raw = row["Bi-Rads"]

        try:
            breast_birads, birads_num = normalize_birads(birads_raw)
        except ValueError:
            invalid_birads.append(str(file_name))
            if args.strict:
                raise
            continue

        image_path = resolve_image_path(file_name, images_dir, image_index)
        if image_path is None:
            missing_files.append(str(file_name))
            if args.strict:
                raise FileNotFoundError(f"Image not found for File Name='{file_name}'")
            continue

        if not is_readable_image(image_path):
            unreadable_files.append(str(image_path))
            if args.strict:
                raise ValueError(f"Unreadable image: {image_path}")
            continue

        label = 1 if birads_num == 1 else 0  # 1 for healthy, 0 for anomalous

        output_rows.append(
            {
                "image_id": os.path.relpath(image_path, images_dir).replace(os.sep, "/"),
                "breast_birads": breast_birads,
                "split": args.split,
                "healthy": label,
                "has_counterfactual": 0,
                "inbreast_file_name": str(file_name),
                "inbreast_bi_rads_raw": str(birads_raw),
            }
        )

    output_df = pd.DataFrame(output_rows)

    # If --output-path is a directory, write a default CSV filename inside it.
    if output_path.suffix.lower() == ".csv":
        csv_output_path = output_path
    else:
        csv_output_path = output_path / "inbreast_test_metadata.csv"

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(csv_output_path, index=False)

    print(f"Saved {len(output_df)} rows to: {csv_output_path}")
    print(f"Missing images skipped: {len(missing_files)}")
    print(f"Unreadable images skipped: {len(unreadable_files)}")
    print(f"Invalid Bi-Rads skipped: {len(invalid_birads)}")

    if missing_files:
        print("First missing File Name entries:", missing_files[:10])
    if unreadable_files:
        print("First unreadable image paths:", unreadable_files[:10])
    if invalid_birads:
        print("First invalid Bi-Rads entries:", invalid_birads[:10])


if __name__ == "__main__":
    main()
