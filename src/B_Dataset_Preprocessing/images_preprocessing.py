# Prepare dataset by running CLAHE on all images
import argparse
import cv2
import os
import shutil
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    CF_DIR,
    DATASET_DIR,
    RAW_IMAGES_DIR,
    REORGANIZED_IMAGES_DIR,
    UPSCALED_CF_DIR,
)

def resize_image_with_aspect_ratio(image, target_size=512):
    """
    Resize an image with aspect ratio preserved and pad (with zeros) to target_size x target_size.
    """
    if image is None:
        raise ValueError("Image cannot be None.")

    # image shape is (height, width) for grayscale
    orig_height, orig_width = image.shape[:2]
    print(f"Original image size: {orig_width}x{orig_height}")

    scale = target_size / max(orig_width, orig_height)
    new_w = int(orig_width * scale)
    new_h = int(orig_height * scale)
    print(f"Resized image size (before padding): {new_w}x{new_h}")

    # Resize the image content using the computed scaled dimensions
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_w
    pad_bottom = target_size - new_h - pad_h
    print(f"Padding added: top={pad_h}, bottom={pad_bottom}, left={pad_w}, right={pad_right}")

    return cv2.copyMakeBorder(
        resized_image,
        pad_h,
        pad_bottom,
        pad_w,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )


def process_dataset(input_dir, output_dir, resize_images, target_size=None, apply_clahe=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if resize_images and target_size is None:
        raise ValueError("`target_size` must be provided when `resize_images` is True")

    image_files = [
        filename
        for filename in os.listdir(input_dir)
        if filename.lower().endswith('.png')
    ]
    
    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at path {input_path} could not be read.")

        if resize_images:
            # Resize the image while preserving aspect ratio and padding to target size
            image = resize_image_with_aspect_ratio(
                image,
                target_size=target_size,
            )

        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

        cv2.imwrite(output_path, image)
        print(f"Processed {filename} and saved to {output_path}")


def reorganize_images_folder(src_dir=None, dest_dir=None):
    """
    Copies image files to a new folder without patient id subfolders.
    """
    src_dir = src_dir or RAW_IMAGES_DIR
    dest_dir = dest_dir or REORGANIZED_IMAGES_DIR

    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory {src_dir} does not exist")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, _, files in os.walk(src_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext != '.png':
                continue

            src_path = os.path.join(root, fname)
            dest_name = fname
            dest_path = os.path.join(dest_dir, dest_name)
            shutil.copy2(src_path, dest_path)

    return dest_dir


def upscale_counterfactuals(original_cf_dir=None, output_cf_dir=None, target_sizes=(912, 1520)):
    """
    Remove padding from generated counterfactual images and resize the cropped content to the target size.
    """
    original_cf_dir = original_cf_dir or CF_DIR
    output_cf_dir = output_cf_dir or UPSCALED_CF_DIR

    if not os.path.exists(original_cf_dir):
        raise ValueError(f"Source directory {original_cf_dir} does not exist")
    if not os.path.exists(output_cf_dir):
        os.makedirs(output_cf_dir)

    target_width, target_height = target_sizes

    image_files = [
        filename
        for filename in os.listdir(original_cf_dir)
        if filename.lower().endswith('.png')
    ]

    for filename in tqdm(image_files, desc="Upscaling counterfactuals"):

        input_path = os.path.join(original_cf_dir, filename)
        output_path = os.path.join(output_cf_dir, filename)

        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at path {input_path} could not be read.")

        corner_size = max(1, min(image.shape[0], image.shape[1]) // 32)
        corner_pixels = [
            image[:corner_size, :corner_size],
            image[:corner_size, -corner_size:],
            image[-corner_size:, :corner_size],
            image[-corner_size:, -corner_size:],
        ]
        background_level = int(min(int(corner.mean()) for corner in corner_pixels))
        non_zero_points = cv2.findNonZero((image > background_level).astype('uint8'))
        if non_zero_points is not None:
            x, y, width, height = cv2.boundingRect(non_zero_points)
            image = image[y:y + height, x:x + width]

        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(output_path, image)
        print(f"Saved resized {filename} to {output_path} with shape {image.shape[1]}x{image.shape[0]}")

# Given a directory with images, calculate and print the mean and std of pixel values across the dataset
def print_mean_std(image_dir):
    pixel_count = 0
    pixel_sum = 0.0
    pixel_sum_sq = 0.0

    image_files = [
        filename
        for filename in os.listdir(image_dir)
        if filename.lower().endswith('.png')
    ]

    for filename in tqdm(image_files, desc="Calculating mean/std"):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_float = image.astype(np.float64)
            pixel_count += image_float.size
            pixel_sum += image_float.sum()
            pixel_sum_sq += np.square(image_float).sum()
        else:
            print(f"Warning: Could not read image {image_path}")

    if pixel_count:
        mean_val = pixel_sum / pixel_count
        variance = (pixel_sum_sq / pixel_count) - (mean_val ** 2)
        std_val = np.sqrt(max(variance, 0.0))
        normalized_mean = mean_val / 255.0
        normalized_std = std_val / 255.0
        print(f"Dataset Mean (0-255): {mean_val:.6f}, Std (0-255): {std_val:.6f}")
        print(f"Dataset Mean (0-1): {normalized_mean:.6f}, Std (0-1): {normalized_std:.6f}")
    else:
        print("No pixel values found to calculate mean and std.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess PNG mammogram images.")
    parser.add_argument(
        "--operation",
        choices=["process", "reorganize", "upscale-counterfactuals", "mean-std"],
        required=True,
        help="Preprocessing operation to run.",
    )
    parser.add_argument("--input-dir", default=None, help="Input image directory.")
    parser.add_argument("--output-dir", default=None, help="Output image directory.")
    parser.add_argument("--resize-images", action="store_true")
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--no-clahe", action="store_true")
    args = parser.parse_args()

    if args.operation == "process":
        process_dataset(
            args.input_dir or REORGANIZED_IMAGES_DIR,
            args.output_dir or DATASET_DIR,
            resize_images=args.resize_images,
            target_size=args.target_size if args.resize_images else None,
            apply_clahe=not args.no_clahe,
        )
    elif args.operation == "reorganize":
        reorganize_images_folder(args.input_dir, args.output_dir)
    elif args.operation == "upscale-counterfactuals":
        upscale_counterfactuals(args.input_dir, args.output_dir)
    else:
        print_mean_std(args.input_dir or DATASET_DIR)