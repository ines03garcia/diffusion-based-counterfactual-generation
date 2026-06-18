import argparse
import ast
import json
import random
import shutil
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image


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
    "4d96d394fe4b20778014bdde96b5d34e",
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
    "dbe631b24f8759c7f7022513582c39ba",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select 70 counterfactual and 30 healthy mammograms. "
            "Save clean images to output_dir and bbox-overlay images to output_bb_dir."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/images/VinDrMammo-CLIP-512-CLAHE",
        help="Root directory for original healthy images",
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
        default="data/images/radiologist_assessment/full_realism",
        help="Output directory for the 100 clean selected images",
    )
    parser.add_argument(
        "--output_bb_dir",
        "--output_cf_dir",
        dest="output_bb_dir",
        type=str,
        default="data/images/radiologist_assessment/healthy_patch_realism",
        help=(
            "Output directory for the same 100 images with bounding boxes drawn. "
            "--output_cf_dir is kept as an alias for backward compatibility."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/metadata/radiologist_dataset.json",
        help="Output JSON file for selected images metadata",
    )
    parser.add_argument(
        "--bbox_line_width",
        type=int,
        default=2,
        help="Line width for drawn bounding boxes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )

    return parser.parse_args()


def image_stem(image_id: str) -> str:
    return Path(image_id).stem


def find_image(image_dir: Path, image_id: str) -> Path:
    """
    Finds an image in image_dir.

    Supports:
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


def parse_coordinate_value(value: Any) -> Optional[list[int]]:
    """
    Parses one bbox coordinate field.

    Handles:
    - [239, 164]
    - "[]"
    - "[239, 164]"
    - 239
    - "239"
    - None
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

        try:
            value = ast.literal_eval(value)
        except Exception:
            try:
                value = int(float(value))
            except Exception:
                return None

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None

        parsed = []
        for item in value:
            if item is None or item == "":
                continue
            try:
                parsed.append(int(float(item)))
            except Exception:
                continue

        return parsed if len(parsed) > 0 else None

    try:
        return [int(float(value))]
    except Exception:
        return None


def get_metadata_bboxes(entry: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    """
    Reads bounding boxes directly from the metadata entry.

    Expected format:
        "resized_xmin": [239, 164],
        "resized_ymin": [212, 231],
        "resized_xmax": [388, 300],
        "resized_ymax": [336, 317]

    Returns a list of boxes:
        [(xmin, ymin, xmax, ymax), ...]
    """
    required_cols = [
        "resized_xmin",
        "resized_ymin",
        "resized_xmax",
        "resized_ymax",
    ]

    if not all(col in entry for col in required_cols):
        return []

    xmin = parse_coordinate_value(entry.get("resized_xmin"))
    ymin = parse_coordinate_value(entry.get("resized_ymin"))
    xmax = parse_coordinate_value(entry.get("resized_xmax"))
    ymax = parse_coordinate_value(entry.get("resized_ymax"))

    if xmin is None or ymin is None or xmax is None or ymax is None:
        return []

    n = min(len(xmin), len(ymin), len(xmax), len(ymax))
    if n == 0:
        return []

    boxes = []

    for i in range(n):
        x1 = int(xmin[i])
        y1 = int(ymin[i])
        x2 = int(xmax[i])
        y2 = int(ymax[i])

        # Skip explicit no-finding boxes.
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue

        # Skip invalid boxes.
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append((x1, y1, x2, y2))

    return boxes


def clamp_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))

    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return x1, y1, x2, y2


def hypothetical_bbox_inside_breast_tissue(img_np: np.ndarray) -> tuple[int, int, int, int]:
    """
    Creates a hypothetical bbox inside the breast tissue.

    Assumes mammogram background is mostly black.
    """
    height, width = img_np.shape

    threshold = max(5, int(np.percentile(img_np, 15)))
    mask = img_np > threshold

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        box_w = max(24, int(width * 0.20))
        box_h = max(24, int(height * 0.20))
        cx = width // 2
        cy = height // 2

        return clamp_box(
            (
                cx - box_w // 2,
                cy - box_h // 2,
                cx + box_w // 2,
                cy + box_h // 2,
            ),
            width,
            height,
        )

    tissue_xmin = int(xs.min())
    tissue_xmax = int(xs.max())
    tissue_ymin = int(ys.min())
    tissue_ymax = int(ys.max())

    tissue_w = tissue_xmax - tissue_xmin + 1
    tissue_h = tissue_ymax - tissue_ymin + 1

    box_w = max(24, int(tissue_w * 0.22))
    box_h = max(24, int(tissue_h * 0.22))

    # Median gives a point usually inside the breast tissue, away from pure background.
    cx = int(np.median(xs))
    cy = int(np.median(ys))

    x1 = cx - box_w // 2
    y1 = cy - box_h // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    return clamp_box((x1, y1, x2, y2), width, height)


def draw_bboxes_on_image(
    image_path: Path,
    output_path: Path,
    entry: dict[str, Any],
    line_width: int = 1,
) -> dict[str, Any]:
    """
    Draws metadata bboxes if available.
    Otherwise draws one hypothetical bbox inside the breast tissue.

    Always saves an image to output_path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Same strategy that you said worked.
    img = Image.open(image_path).convert("L")
    img_np = np.array(img)

    height, width = img_np.shape

    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    metadata_boxes = get_metadata_bboxes(entry)

    if len(metadata_boxes) > 0:
        boxes = [clamp_box(box, width, height) for box in metadata_boxes]
        bbox_type = "metadata_resized_coordinates"
    else:
        boxes = [hypothetical_bbox_inside_breast_tissue(img_np)]
        bbox_type = "hypothetical_breast_tissue"

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(
            img_rgb,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            line_width,
        )

    img_with_bbox = Image.fromarray(img_rgb)
    img_with_bbox.save(output_path)

    return {
        "bbox_image_path": str(output_path),
        "bbox_type": bbox_type,
        "bboxes_drawn": [list(box) for box in boxes],
    }


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    cf_dir = Path(args.cf_dir)
    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)
    output_bb_dir = Path(args.output_bb_dir)
    output_json = Path(args.output_json)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_bb_dir.mkdir(parents=True, exist_ok=True)
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

    healthy_entries = [
        entry
        for entry in metadata
        if entry.get("healthy") == 1 and entry.get("split") == "test"
    ]

    if len(cf_entries) < 70:
        raise ValueError(
            f"Need 70 counterfactual entries, but found only {len(cf_entries)}"
        )

    if len(healthy_entries) < 30:
        raise ValueError(
            f"Need 30 healthy entries, but found only {len(healthy_entries)}"
        )

    selected_cf = random.sample(cf_entries, 70)
    selected_healthy = random.sample(healthy_entries, 30)

    selected_metadata = []

    metadata_boxes_count = 0
    hypothetical_boxes_count = 0

    # 1. Counterfactuals:
    #    Clean image goes to output_dir.
    #    Same image with bbox goes to output_bb_dir.
    for entry in selected_cf:
        image_id = entry["image_id"]

        src_path = find_image(cf_dir, image_id)

        clean_dst_path = output_dir / image_id
        bbox_dst_path = output_bb_dir / image_id

        clean_dst_path.parent.mkdir(parents=True, exist_ok=True)
        bbox_dst_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_path, clean_dst_path)

        new_entry = dict(entry)
        new_entry["source"] = "counterfactual"
        new_entry["copied_from"] = str(src_path)
        new_entry["clean_copied_to"] = str(clean_dst_path)

        bbox_info = draw_bboxes_on_image(
            image_path=clean_dst_path,
            output_path=bbox_dst_path,
            entry=new_entry,
            line_width=args.bbox_line_width,
        )
        new_entry.update(bbox_info)

        if bbox_info["bbox_type"] == "metadata_resized_coordinates":
            metadata_boxes_count += 1
        else:
            hypothetical_boxes_count += 1

        selected_metadata.append(new_entry)

    # 2. Healthy originals:
    #    Clean image goes to output_dir.
    #    Same image with bbox goes to output_bb_dir.
    for entry in selected_healthy:
        image_id = entry["image_id"]

        src_path = find_image(data_dir, image_id)

        clean_dst_path = output_dir / image_id
        bbox_dst_path = output_bb_dir / image_id

        clean_dst_path.parent.mkdir(parents=True, exist_ok=True)
        bbox_dst_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_path, clean_dst_path)

        new_entry = dict(entry)
        new_entry["source"] = "healthy_original"
        new_entry["copied_from"] = str(src_path)
        new_entry["clean_copied_to"] = str(clean_dst_path)

        bbox_info = draw_bboxes_on_image(
            image_path=clean_dst_path,
            output_path=bbox_dst_path,
            entry=new_entry,
            line_width=args.bbox_line_width,
        )
        new_entry.update(bbox_info)

        if bbox_info["bbox_type"] == "metadata_resized_coordinates":
            metadata_boxes_count += 1
        else:
            hypothetical_boxes_count += 1

        selected_metadata.append(new_entry)

    with open(output_json, "w") as f:
        json.dump(selected_metadata, f, indent=2)

    # Hard checks: fail loudly if anything is missing.
    clean_missing = [
        entry["clean_copied_to"]
        for entry in selected_metadata
        if not Path(entry["clean_copied_to"]).exists()
    ]

    bbox_missing = [
        entry["bbox_image_path"]
        for entry in selected_metadata
        if not Path(entry["bbox_image_path"]).exists()
    ]

    if clean_missing:
        raise RuntimeError(
            f"{len(clean_missing)} clean images are missing. "
            f"First missing file: {clean_missing[0]}"
        )

    if bbox_missing:
        raise RuntimeError(
            f"{len(bbox_missing)} bbox images are missing. "
            f"First missing file: {bbox_missing[0]}"
        )

    print(f"Copied {len(selected_cf)} clean counterfactual images to {output_dir}")
    print(f"Copied {len(selected_healthy)} clean healthy images to {output_dir}")
    print(f"Saved {len(selected_metadata)} bbox-overlay images to {output_bb_dir}")
    print(f"Metadata bbox images: {metadata_boxes_count}")
    print(f"Hypothetical bbox images: {hypothetical_boxes_count}")
    print(f"Saved metadata JSON to {output_json}")


if __name__ == "__main__":
    main()