import os
import json
from PIL import Image
import argparse
import numpy as np


def get_largest_bbox_from_resized_coords(item):
    """
    Return the largest bbox (xmin, ymin, xmax, ymax)
    """
    xmins = item.get("resized_xmin", [])
    ymins = item.get("resized_ymin", [])
    xmaxs = item.get("resized_xmax", [])
    ymaxs = item.get("resized_ymax", [])

    n_boxes = min(len(xmins), len(ymins), len(xmaxs), len(ymaxs))

    if n_boxes == 0:
        return None

    largest_bbox = None
    largest_area = 0

    for i in range(n_boxes):
        xmin = float(xmins[i])
        ymin = float(ymins[i])
        xmax = float(xmaxs[i])
        ymax = float(ymaxs[i])

        width = xmax - xmin
        height = ymax - ymin

        if width <= 0 or height <= 0:
            continue

        area = width * height

        if area > largest_area:
            largest_area = area
            largest_bbox = (xmin, ymin, xmax, ymax)

    if largest_bbox is None:
        print(f"{xmins}, {ymins}, {xmaxs}, {ymaxs}")
    return largest_bbox


def crop_bbox_xyxy(image, bbox):
    """
    Crop PIL image using bbox in (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = bbox
    img_w, img_h = image.size

    x1 = max(0, min(int(round(xmin)), img_w))
    y1 = max(0, min(int(round(ymin)), img_h))
    x2 = max(0, min(int(round(xmax)), img_w))
    y2 = max(0, min(int(round(ymax)), img_h))

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))

def get_healthy_patch(metadata_path, data_dir, bbox, laterality, view, healthy_output_dir):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    already_saved_healthy_ids = set(os.listdir(healthy_output_dir))

    for item in metadata:
        if item.get("healthy") == 1 and item.get("laterality") == laterality and item.get("view") == view and item.get("image_id") not in already_saved_healthy_ids:
            img = Image.open(os.path.join(data_dir, item.get("image_id"))).convert("L")
            img = crop_bbox_xyxy(img, bbox)
            img = crop_until_no_black_boundary(img, threshold=5)
            return img, item.get("image_id")
    return None, None


def crop_counterfactuals_to_largest_bbox(
    metadata_path="data/metadata/processed_df_birads.json",
    cf_dir=None,
    output_dir=None,
    data_dir=None,
    healthy_output_dir="/home/csantiago/inescgarcia/diffusion-based-counterfactual-generation/data/images/healthy_patches_for_fid"
):
    if cf_dir is None:
        raise ValueError("cf_dir must be provided.")
    if output_dir is None:
        raise ValueError("output_dir must be provided.")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(healthy_output_dir, exist_ok=True)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    saved_paths = []
    skipped_no_bbox = 0
    skipped_missing_image = 0
    skipped_invalid_bbox = 0

    for item in metadata:
        if item.get("has_cf") != 1: # For every metadata row with has_cf == 1
            continue

        image_id = item.get("image_id")
        if image_id is None:
            print("Skipping row without image_id.")
            continue

        bbox = get_largest_bbox_from_resized_coords(item)

        if bbox is None:
            print(f"No valid bbox found for {image_id}")
            skipped_no_bbox += 1
            continue

        cf_image_path = os.path.join(cf_dir, image_id) # Load the corresponding counterfactual image from cf_dir

        if not os.path.exists(cf_image_path):
            print(f"Counterfactual image not found: {cf_image_path}")
            skipped_missing_image += 1
            continue

        try:
            image = Image.open(cf_image_path).convert("L")
        except Exception as e:
            print(f"Could not open {cf_image_path}: {e}")
            continue

        cropped = crop_bbox_xyxy(image, bbox) # Crop the counterfactual image to the bbox with the biggest area

        if cropped is None:
            print(f"Invalid bbox {bbox} for {image_id}")
            skipped_invalid_bbox += 1
            continue

        cropped = crop_until_no_black_boundary(cropped, threshold=5)

        output_path = os.path.join(output_dir, image_id)

        cropped.save(output_path)
        saved_paths.append(output_path)

        healthy_patch, healthy_id = get_healthy_patch(metadata_path, data_dir, bbox, item.get("laterality"), item.get("view"), healthy_output_dir)
        if healthy_patch is not None:
            healthy_patch.save(os.path.join(healthy_output_dir, healthy_id))
            print(f"Saved healthy patch for {healthy_id} corresponding to {image_id} at {os.path.join(healthy_output_dir, healthy_id)}")
        else:
            print(f"No healthy patch found for {image_id} with laterality {item.get('laterality')} and view {item.get('view')}")

    print(f"Saved {len(saved_paths)} cropped counterfactual patches to {output_dir}")
    print(f"Skipped without bbox: {skipped_no_bbox}")
    print(f"Skipped missing image: {skipped_missing_image}")
    print(f"Skipped invalid bbox: {skipped_invalid_bbox}")

    return saved_paths


def crop_until_no_black_boundary(image, threshold=5, max_iters=None):
    """
    Iteratively crop image boundaries until no boundary pixels are bellow the threshold.
    """
    arr = np.asarray(image)

    if arr.ndim != 2:
        raise ValueError("Expected a grayscale image.")

    y1 = 0
    y2 = arr.shape[0]
    x1 = 0
    x2 = arr.shape[1]

    iters = 0

    while y2 > y1 and x2 > x1:
        cropped = arr[y1:y2, x1:x2]

        changed = False

        # Top row
        if np.any(cropped[0, :] <= threshold):
            y1 += 1
            changed = True

        # Bottom row
        cropped = arr[y1:y2, x1:x2]
        if cropped.shape[0] > 0 and np.any(cropped[-1, :] <= threshold):
            y2 -= 1
            changed = True

        # Left column
        cropped = arr[y1:y2, x1:x2]
        if cropped.shape[1] > 0 and np.any(cropped[:, 0] <= threshold):
            x1 += 1
            changed = True

        # Right column
        cropped = arr[y1:y2, x1:x2]
        if cropped.shape[1] > 0 and np.any(cropped[:, -1] <= threshold):
            x2 -= 1
            changed = True

        if not changed:
            break

        iters += 1
        if max_iters is not None and iters >= max_iters:
            break

    if y2 <= y1 or x2 <= x1:
        return image

    return image.crop((x1, y1, x2, y2))

def main():
    parser = argparse.ArgumentParser(description="Crop counterfactuals and original images in patches and save them in folders for FID calculation.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata JSON file.")
    parser.add_argument("--cf_dir", type=str, required=True, help="Directory containing counterfactual images.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing original images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where cropped counterfactual patches will be saved.")

    args = parser.parse_args()

    saved_paths = crop_counterfactuals_to_largest_bbox(
        metadata_path=args.metadata_path,
        cf_dir=args.cf_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    print(f"Saved {len(saved_paths)} cropped counterfactual patches to:")
    print(args.output_dir)


if __name__ == "__main__":
    main()