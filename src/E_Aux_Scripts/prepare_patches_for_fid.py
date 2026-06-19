import os
import json
from PIL import Image
import argparse
import numpy as np


def get_largest_bbox_from_resized_coords(item):
    """
    Return the largest bbox as (xmin, ymin, xmax, ymax).
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
        print(f"Invalid bbox coords: {xmins}, {ymins}, {xmaxs}, {ymaxs}")

    return largest_bbox


def crop_bbox_xyxy(image, bbox):
    """
    Crop PIL image using bbox in (xmin, ymin, xmax, ymax).
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


def crop_until_no_black_boundary(image, threshold=5, max_iters=None):
    """
    Iteratively crop image boundaries until no boundary pixels are below/equal to threshold.
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
        if cropped.shape[0] > 0 and np.any(cropped[0, :] <= threshold):
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


def is_patch_large_enough(image, min_patch_size=32):
    """
    Check size before resizing.

    Returns True only if both width and height are at least min_patch_size.
    """
    if image is None:
        return False

    width, height = image.size
    return width >= min_patch_size and height >= min_patch_size


def patch_black_fraction(image, black_threshold=5):
    """
    Fraction of pixels considered black.
    """
    arr = np.asarray(image)

    if arr.size == 0:
        return 1.0

    return float(np.mean(arr <= black_threshold))


def is_patch_black_or_constant(image, black_threshold=5, max_black_fraction=0.0):
    """
    Returns (is_bad, reason).

    A bad patch is:
    - empty
    - constant
    - has black pixel fraction above max_black_fraction
    """
    if image is None:
        return True, "patch_is_none"

    arr = np.asarray(image)

    if arr.size == 0:
        return True, "empty_patch"

    if arr.min() == arr.max():
        return True, f"constant_patch_value_{arr.min()}"

    black_frac = patch_black_fraction(image, black_threshold=black_threshold)

    if black_frac > max_black_fraction:
        return True, f"too_black_fraction_{black_frac:.4f}"

    return False, "valid"


def get_healthy_patch(
    metadata,
    data_dir,
    bbox,
    laterality,
    view,
    healthy_output_dir,
    black_threshold=5,
    max_black_fraction=0.0,
):
    """
    Find a healthy image with matching laterality/view and crop it using the same bbox.

    This function rejects black/constant healthy patches.

    It does NOT reject small patches here. Size is checked in the main pair logic,
    so if the healthy patch is too small, the whole CF/healthy pair is skipped.
    """
    already_saved_healthy_ids = set(os.listdir(healthy_output_dir))

    skipped_missing = 0
    skipped_invalid_crop = 0
    skipped_black_or_constant = 0

    for item in metadata:
        image_id = item.get("image_id")

        if (
            item.get("healthy") != 1
            or item.get("laterality") != laterality
            or item.get("view") != view
            or image_id in already_saved_healthy_ids
        ):
            continue

        image_path = os.path.join(data_dir, image_id)

        if not os.path.exists(image_path):
            skipped_missing += 1
            continue

        try:
            img = Image.open(image_path).convert("L")
        except Exception as e:
            print(f"Could not open healthy image {image_path}: {e}")
            skipped_missing += 1
            continue

        patch = crop_bbox_xyxy(img, bbox)

        if patch is None:
            skipped_invalid_crop += 1
            continue

        patch = crop_until_no_black_boundary(
            patch,
            threshold=black_threshold,
        )

        is_bad, reason = is_patch_black_or_constant(
            patch,
            black_threshold=black_threshold,
            max_black_fraction=max_black_fraction,
        )

        if is_bad:
            skipped_black_or_constant += 1
            print(f"Skipping healthy candidate {image_id}: {reason}")
            continue

        return patch, image_id

    print(
        "No valid healthy patch found "
        f"for laterality={laterality}, view={view}. "
        f"Skipped missing={skipped_missing}, "
        f"invalid_crop={skipped_invalid_crop}, "
        f"black_or_constant={skipped_black_or_constant}"
    )

    return None, None


def crop_counterfactuals_to_largest_bbox(
    metadata_path="data/metadata/processed_df_birads.json",
    cf_dir=None,
    output_dir=None,
    data_dir=None,
    healthy_output_dir="data/images/patches_for_fid/healthy_patches/healthy_patches_for_fid_512",
    min_patch_size=32,
    black_threshold=5,
    max_black_fraction=0.0,
    resize_size=512,
):
    if cf_dir is None:
        raise ValueError("cf_dir must be provided.")

    if output_dir is None:
        raise ValueError("output_dir must be provided.")

    if data_dir is None:
        raise ValueError("data_dir must be provided.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(healthy_output_dir, exist_ok=True)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    saved_cf_paths = []
    saved_healthy_paths = []

    skipped_no_bbox = 0
    skipped_missing_cf_image = 0
    skipped_invalid_bbox = 0
    skipped_cf_too_small = 0
    skipped_cf_black_or_constant = 0
    skipped_no_valid_healthy = 0
    skipped_healthy_too_small = 0

    for item in metadata:
        if item.get("has_cf") != 1:
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

        cf_image_path = os.path.join(cf_dir, image_id)

        if not os.path.exists(cf_image_path):
            print(f"Counterfactual image not found: {cf_image_path}")
            skipped_missing_cf_image += 1
            continue

        try:
            cf_image = Image.open(cf_image_path).convert("L")
        except Exception as e:
            print(f"Could not open {cf_image_path}: {e}")
            skipped_missing_cf_image += 1
            continue

        # 1. Crop CF patch first.
        cf_patch = crop_bbox_xyxy(cf_image, bbox)

        if cf_patch is None:
            print(f"Invalid bbox {bbox} for {image_id}")
            skipped_invalid_bbox += 1
            continue

        # 2. Remove black boundary from CF patch.
        cf_patch = crop_until_no_black_boundary(
            cf_patch,
            threshold=black_threshold,
        )

        # 3. Reject CF if black/constant.
        cf_is_bad, cf_reason = is_patch_black_or_constant(
            cf_patch,
            black_threshold=black_threshold,
            max_black_fraction=max_black_fraction,
        )

        if cf_is_bad:
            print(f"Skipping {image_id}: counterfactual patch is invalid: {cf_reason}")
            skipped_cf_black_or_constant += 1
            continue

        # 4. Reject CF if too small BEFORE resizing.
        if not is_patch_large_enough(cf_patch, min_patch_size=min_patch_size):
            w, h = cf_patch.size
            print(
                f"Skipping {image_id}: counterfactual patch too small "
                f"after black-boundary crop ({w}x{h})"
            )
            skipped_cf_too_small += 1
            continue

        # 5. Only now sample a matching healthy patch.
        healthy_patch, healthy_id = get_healthy_patch(
            metadata=metadata,
            data_dir=data_dir,
            bbox=bbox,
            laterality=item.get("laterality"),
            view=item.get("view"),
            healthy_output_dir=healthy_output_dir,
            black_threshold=black_threshold,
            max_black_fraction=max_black_fraction,
        )

        if healthy_patch is None:
            print(
                f"No valid healthy patch found for {image_id} "
                f"with laterality={item.get('laterality')}, view={item.get('view')}"
            )
            skipped_no_valid_healthy += 1
            continue

        # 6. Reject the pair if healthy patch is too small BEFORE resizing.
        if not is_patch_large_enough(healthy_patch, min_patch_size=min_patch_size):
            w, h = healthy_patch.size
            print(
                f"Skipping pair for {image_id}: healthy patch {healthy_id} "
                f"too small after black-boundary crop ({w}x{h})"
            )
            skipped_healthy_too_small += 1
            continue

        # 7. Save both only after both are valid.
        cf_patch = cf_patch.resize((resize_size, resize_size), Image.BICUBIC)
        healthy_patch = healthy_patch.resize((resize_size, resize_size), Image.BICUBIC)

        cf_output_path = os.path.join(output_dir, image_id)
        healthy_output_path = os.path.join(healthy_output_dir, healthy_id)

        cf_patch.save(cf_output_path)
        healthy_patch.save(healthy_output_path)

        saved_cf_paths.append(cf_output_path)
        saved_healthy_paths.append(healthy_output_path)

        print(f"Saved CF patch: {cf_output_path}")
        print(
            f"Saved healthy patch for {healthy_id} "
            f"corresponding to {image_id}: {healthy_output_path}"
        )

    print()
    print("Summary")
    print("-------")
    print(f"Saved CF patches: {len(saved_cf_paths)}")
    print(f"Saved healthy patches: {len(saved_healthy_paths)}")
    print(f"Skipped without bbox: {skipped_no_bbox}")
    print(f"Skipped missing CF image: {skipped_missing_cf_image}")
    print(f"Skipped invalid bbox: {skipped_invalid_bbox}")
    print(f"Skipped CF black/constant: {skipped_cf_black_or_constant}")
    print(f"Skipped CF too small: {skipped_cf_too_small}")
    print(f"Skipped no valid healthy: {skipped_no_valid_healthy}")
    print(f"Skipped healthy too small: {skipped_healthy_too_small}")

    return saved_cf_paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crop counterfactual and healthy image patches for FID/FRD calculation. "
            "Pairs are saved only if both CF and healthy patches are valid."
        )
    )

    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/metadata/processed_df_birads.json",
        help="Path to the metadata JSON file.",
    )
    parser.add_argument(
        "--cf_dir",
        type=str,
        default="data/images/repaint_results_912x1520",
        help="Directory containing counterfactual images.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/images/VinDrMammo-CLIP-512-CLAHE-912-1520",
        help="Directory containing original images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/images/patches_for_fid/cf_patches/repaint_patches_512",
        help="Directory where cropped counterfactual patches will be saved.",
    )
    parser.add_argument(
        "--healthy_output_dir",
        type=str,
        default="data/images/patches_for_fid/healthy_patches/healthy_patches_for_fid_512",
        help="Directory where cropped healthy patches will be saved.",
    )
    parser.add_argument(
        "--min_patch_size",
        type=int,
        default=32,
        help="Minimum width and height allowed for both CF and healthy patches before resizing.",
    )
    parser.add_argument(
        "--black_threshold",
        type=int,
        default=5,
        help="Pixels <= this value are considered black.",
    )
    parser.add_argument(
        "--max_black_fraction",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed fraction of black pixels in a patch. "
            "Use 0.0 to reject any patch containing black pixels."
        ),
    )
    parser.add_argument(
        "--resize_size",
        type=int,
        default=512,
        help="Final square patch size after validation.",
    )

    args = parser.parse_args()

    saved_paths = crop_counterfactuals_to_largest_bbox(
        metadata_path=args.metadata_path,
        cf_dir=args.cf_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        healthy_output_dir=args.healthy_output_dir,
        min_patch_size=args.min_patch_size,
        black_threshold=args.black_threshold,
        max_black_fraction=args.max_black_fraction,
        resize_size=args.resize_size,
    )

    print(f"Saved {len(saved_paths)} cropped counterfactual patches to:")
    print(args.output_dir)


if __name__ == "__main__":
    main()