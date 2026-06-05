# Prepare dataset by running CLAHE on all images
import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm

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
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png')):
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


def reorganize_images_folder(src_dir="data/images/images_png", dest_dir="data/images/VinDr-Mammo-CLIP"):
    """
    Copies image files to a new folder without patient id subfolders.
    """
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


def upscale_counterfactuals(original_cf_dir="data/images/repaint_results", output_cf_dir="data/images/repaint_results_912x1520", target_sizes=(912, 1520)):
    """
    Remove padding from generated counterfactual images and resize the cropped content to the target size.
    """
    if not os.path.exists(original_cf_dir):
        raise ValueError(f"Source directory {original_cf_dir} does not exist")
    if not os.path.exists(output_cf_dir):
        os.makedirs(output_cf_dir)

    target_width, target_height = target_sizes

    for filename in os.listdir(original_cf_dir):
        if not filename.lower().endswith('.png'):
            continue

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


if __name__ == "__main__":
    input_directory = "data/images/VinDrMammo-CLIP" # Input images directory
    output_directory = "data/images/VinDrMammo-CLIP-CLAHE" # Output directory for resized CLAHE images

    #input_directory = reorganize_images_folder(src_dir="data/images/images_png", dest_dir="data/images/VinDrMammo-CLIP")
    
    #process_dataset(input_directory, output_directory, resize_images=False, apply_clahe=True)

    #upscale_counterfactuals(original_cf_dir="data/images/repaint_results", output_cf_dir="data/images/repaint_results_912x1520", target_sizes=(912, 1520))

    print_mean_std(output_directory)