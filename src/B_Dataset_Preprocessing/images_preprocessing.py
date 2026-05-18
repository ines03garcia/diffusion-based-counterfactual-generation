# Prepare dataset by running CLAHE on all images
import cv2
import os

def resize_image_with_aspect_ratio(image, orig_width=None, orig_height=None, target_size=512):
    """Resize an image using the same scale-and-pad logic as bbox resizing.

    If orig_width/orig_height are not provided, they are taken from the image itself.
    """
    if image is None:
        raise ValueError("Image cannot be None.")

    # image shape is (height, width) for grayscale
    img_h, img_w = image.shape[:2]
    if orig_width is None:
        orig_width = img_w
    if orig_height is None:
        orig_height = img_h

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

def apply_clahe_to_image(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be read.")
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE to the image
    clahe_image = clahe.apply(image)
    
    # Save the processed image
    cv2.imwrite(output_path, clahe_image)


def process_dataset(input_dir, output_dir, resize_images, orig_width=None, orig_height=None, target_size=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if resize_images:
        if target_size is None:
            raise ValueError("`target_size` must be provided when `resize_images` is True")
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Image at path {input_path} could not be read.")

            if resize_images:
                # Use the actual image dimensions so the letterbox matches bbox scaling.
                # If orig_width/orig_height are None they will be inferred inside the helper.
                image = resize_image_with_aspect_ratio(
                    image,
                    orig_width=orig_width,
                    orig_height=orig_height,
                    target_size=target_size,
                )

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(image)
            cv2.imwrite(output_path, clahe_image)
            print(f"Processed {filename} and saved to {output_path}")


if __name__ == "__main__":
    input_directory = "data/images/VinDrMammo-CLIP" # Input images directory
    output_directory = "data/images/VinDrMammo-CLIP2" # Output directory for resized CLAHE images
    # Resize images before CLAHE. orig_width/orig_height=None lets the helper infer from each image.
    process_dataset(input_directory, output_directory, resize_images=True, orig_width=None, orig_height=None, target_size=512)