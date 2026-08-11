import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path
import ast
from src.config import ROOT

def parse_bbox_list(bbox_str):
    """Parse bounding box coordinate string to list of integers."""
    try:
        bbox_list = ast.literal_eval(bbox_str)
        return bbox_list if isinstance(bbox_list, list) else []
    except:
        return []

def draw_bounding_boxes(image_path, bboxes, output_path, color=(0, 0, 0), thickness=2):
    """
    Draw bounding boxes on an image and save it.
    
    Args:
        image_path: Path to the input image
        bboxes: List of tuples (xmin, ymin, xmax, ymax)
        output_path: Path to save the output image
        color: BGR color tuple for the bounding box (default: black)
        thickness: Line thickness for the bounding box
    """
    # Read the image
    img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return False
    
    # Draw each bounding box
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        # Ensure coordinates are integers
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # Draw rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    
    # Save the image
    cv2.imwrite(str(output_path), img)
    return True

def get_biggest_bbox(bboxes):
    """
    Find the biggest bounding box by area.
    
    Args:
        bboxes: List of tuples (xmin, ymin, xmax, ymax)
    
    Returns:
        Tuple (xmin, ymin, xmax, ymax) of the biggest bbox
    """
    if not bboxes:
        return None
    
    max_area = 0
    biggest_bbox = None
    
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        area = (xmax - xmin) * (ymax - ymin)
        if area > max_area:
            max_area = area
            biggest_bbox = bbox
    
    return biggest_bbox

def crop_bbox_region(image_path, bbox, output_path):
    """
    Crop the region defined by a bounding box and save it.
    
    Args:
        image_path: Path to the input image
        bbox: Tuple (xmin, ymin, xmax, ymax)
        output_path: Path to save the cropped image
    
    Returns:
        True if successful, False otherwise
    """
    # Read the image
    img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return False
    
    xmin, ymin, xmax, ymax = bbox
    # Ensure coordinates are integers
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    
    # Crop the region
    cropped_img = img[ymin:ymax, xmin:xmax]
    
    # Save the cropped image
    cv2.imwrite(str(output_path), cropped_img)
    return True

def main():
    # Paths
    base_dir = Path(ROOT)
    metadata_path = base_dir / "data/metadata/resized_df_has_counterfactual.csv"
    original_images_dir = base_dir / "data/images/VinDr-Mammo-Clip-CLAHE-512"
    counterfactual_images_dir = base_dir / "data/images/repaint_results"
    
    # Output directories
    output_base = base_dir / "data/images/inpaint_results"
    output_original = output_base / "original_bb"
    output_counterfactual = output_base / "counterfactual_bb"
    output_original_closeup = output_base / "original_bb_closeup"
    output_counterfactual_closeup = output_base / "counterfactual_bb_closeup"
    
    # Create output directories
    output_original.mkdir(parents=True, exist_ok=True)
    output_counterfactual.mkdir(parents=True, exist_ok=True)
    output_original_closeup.mkdir(parents=True, exist_ok=True)
    output_counterfactual_closeup.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories:")
    print(f"  - {output_original}")
    print(f"  - {output_counterfactual}")
    print(f"  - {output_original_closeup}")
    print(f"  - {output_counterfactual_closeup}")
    
    # Read metadata
    print(f"\nReading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    # Filter by has_counterfactual == 1
    df_filtered = df[df['has_counterfactual'] == 1].copy()
    print(f"Found {len(df_filtered)} images with counterfactuals")
    
    # Process each image
    success_count = 0
    failed_count = 0
    
    for idx, row in df_filtered.iterrows():
        image_id = row['image_id']
        
        # Parse bounding box coordinates
        xmin_list = parse_bbox_list(row['resized_xmin'])
        ymin_list = parse_bbox_list(row['resized_ymin'])
        xmax_list = parse_bbox_list(row['resized_xmax'])
        ymax_list = parse_bbox_list(row['resized_ymax'])
        
        # Create list of bounding boxes
        bboxes = []
        if len(xmin_list) == len(ymin_list) == len(xmax_list) == len(ymax_list):
            for i in range(len(xmin_list)):
                bboxes.append((xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]))
        
        if not bboxes:
            print(f"Skipping {image_id}: No valid bounding boxes found")
            failed_count += 1
            continue
        
        # Process original image
        original_path = original_images_dir / image_id
        output_original_path = output_original / image_id
        
        if original_path.exists():
            if draw_bounding_boxes(original_path, bboxes, output_original_path):
                print(f"✓ Processed original: {image_id} ({len(bboxes)} boxes)")
            else:
                print(f"✗ Failed to process original: {image_id}")
                failed_count += 1
                continue
        else:
            print(f"✗ Original image not found: {original_path}")
            failed_count += 1
            continue
        
        # Process counterfactual image
        counterfactual_path = counterfactual_images_dir / image_id
        output_counterfactual_path = output_counterfactual / image_id
        
        if counterfactual_path.exists():
            if draw_bounding_boxes(counterfactual_path, bboxes, output_counterfactual_path):
                print(f"✓ Processed counterfactual: {image_id} ({len(bboxes)} boxes)")
            else:
                print(f"✗ Failed to process counterfactual: {image_id}")
                failed_count += 1
                continue
        else:
            print(f"✗ Counterfactual image not found: {counterfactual_path}")
            failed_count += 1
            continue
        
        # Find and crop biggest bounding box for close-up
        biggest_bbox = get_biggest_bbox(bboxes)
        if biggest_bbox:
            # Crop original image
            output_original_closeup_path = output_original_closeup / image_id
            if crop_bbox_region(original_path, biggest_bbox, output_original_closeup_path):
                print(f"✓ Saved original close-up: {image_id}")
            else:
                print(f"✗ Failed to save original close-up: {image_id}")
            
            # Crop counterfactual image
            output_counterfactual_closeup_path = output_counterfactual_closeup / image_id
            if crop_bbox_region(counterfactual_path, biggest_bbox, output_counterfactual_closeup_path):
                print(f"✓ Saved counterfactual close-up: {image_id}")
            else:
                print(f"✗ Failed to save counterfactual close-up: {image_id}")
        
        # Count as success if we got here
        success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count} image pairs")
    print(f"Failed: {failed_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
