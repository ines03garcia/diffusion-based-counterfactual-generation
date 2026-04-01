# Prepare dataset by running CLAHE on all images
import cv2
import os
from src.config import DATASET_DIR, IMAGES_ROOT

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


def process_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            apply_clahe_to_image(input_path, output_path)
            print(f"Processed {filename} and saved to {output_path}")


if __name__ == "__main__":
    input_directory = DATASET_DIR
    output_directory = os.path.join(IMAGES_ROOT, "VinDr-Mammo-Clip-CLAHE-512")
    process_dataset(input_directory, output_directory)