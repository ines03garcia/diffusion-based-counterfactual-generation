import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

"""
Smooth edges of counterfactual images using seamless cloning at bounding box boundaries.
"""
from src.config import IMAGES_ROOT, METADATA_ROOT

# Paths
COUNTERFACTUALS_DIR = os.path.join(IMAGES_ROOT, "counterfactuals_updated")
METADATA_PATH = os.path.join(METADATA_ROOT, "resized_annotations_512.csv")
OUTPUT_DIR = os.path.join(IMAGES_ROOT, "counterfactuals_smoothed")

def load_annotations(metadata_path):
    """Load and parse annotations from CSV."""
    df = pd.read_csv(metadata_path)
    
    # Parse string representations of lists
    for col in ['resized_xmin', 'resized_ymin', 'resized_xmax', 'resized_ymax']:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    return df


def create_mask_for_bbox(image_shape, bbox, margin=15):
    """
    Create a mask for seamless cloning.
    The mask covers the bbox with a margin for smooth blending.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x_min, y_min, x_max, y_max = bbox
    
    # Expand bbox by margin for smoother transition
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)
    
    # Fill the mask region
    mask[y_min:y_max, x_min:x_max] = 255
    
    return mask, (x_min, y_min, x_max, y_max)


def apply_seamless_clone(src_image, dst_image, bbox, margin=15):
    """
    Apply seamless cloning to blend the bbox region from src to dst.
    """
    # Create mask for this bbox
    mask, (x_min, y_min, x_max, y_max) = create_mask_for_bbox(src_image.shape, bbox, margin)
    
    # Calculate center point for seamless clone
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    center = (center_x, center_y)
    
    # Apply seamless cloning (MIXED_CLONE for best blending)
    try:
        result = cv2.seamlessClone(src_image, dst_image, mask, center, cv2.MIXED_CLONE)
        return result
    except cv2.error as e:
        print(f"Warning: Seamless clone failed for bbox {bbox}: {e}")
        return dst_image


def smooth_image_edges(image_path, annotations_df, output_path, margin=15):
    """
    Smooth edges of an image at bounding box boundaries using seamless cloning.
    """
    # Extract image name
    image_name = os.path.basename(image_path)
    
    # Find annotations for this image
    image_row = annotations_df[annotations_df['image_id'] == image_name]
    
    if image_row.empty:
        print(f"No annotations found for {image_name}, copying as-is")
        img = Image.open(image_path)
        img.save(output_path)
        return
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {image_path}")
        return
    
    # Convert to 3-channel for seamless clone (required by OpenCV)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = img_color.copy()
    
    # Get all bounding boxes for this image
    row = image_row.iloc[0]
    nr_anomalies = len(row['resized_xmin'])
    
    # Apply seamless cloning for each bbox
    for i in range(nr_anomalies):
        x_min = int(row['resized_xmin'][i])
        y_min = int(row['resized_ymin'][i])
        x_max = int(row['resized_xmax'][i])
        y_max = int(row['resized_ymax'][i])
        
        bbox = (x_min, y_min, x_max, y_max)
        result = apply_seamless_clone(img_color, result, bbox, margin)
    
    # Convert back to grayscale
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Save result
    cv2.imwrite(output_path, result_gray)


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load annotations
    print("Loading annotations...")
    annotations_df = load_annotations(METADATA_PATH)
    
    # Get all counterfactual images
    image_files = [f for f in os.listdir(COUNTERFACTUALS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for image_file in tqdm(image_files, desc="Smoothing edges"):
        input_path = os.path.join(COUNTERFACTUALS_DIR, image_file)
        output_path = os.path.join(OUTPUT_DIR, image_file)
        
        smooth_image_edges(input_path, annotations_df, output_path, margin=15)
    
    print(f"Done! Smoothed images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()