# Prepare dataset by running CLAHE on all images
import cv2
import os
import shutil

def resize_image_with_aspect_ratio(image, target_size=512):
    """
    Resize an image with aspect ratio preserved and pad (with zeros) to target_size x target_size.
    """
    if image is None:
        raise ValueError("Image cannot be None.")

    # image shape is (height, width) for grayscale
    orig_height, orig_width = image.shape[:2]

    scale = target_size / max(orig_width, orig_height)
    new_w = int(orig_width * scale)
    new_h = int(orig_height * scale)

    # Resize the image content using the computed scaled dimensions
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_w
    pad_bottom = target_size - new_h - pad_h

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


if __name__ == "__main__":
    input_directory = "data/images/VinDrMammo-CLIP" # Input images directory
    output_directory = "data/images/VinDrMammo-CLIP-preprocessed" # Output directory for resized CLAHE images

    input_directory = reorganize_images_folder(src_dir="data/images/images_png", dest_dir="data/images/VinDrMammo-CLIP")
    
    process_dataset(input_directory, output_directory, resize_images=False, apply_clahe=False)