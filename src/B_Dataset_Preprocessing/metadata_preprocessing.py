"""
Preprocess metadata:
1. Drop unnecessary columns and rename "training" split to "train"
2. Validate bounding boxes coordinates
3. (Optional) Resize to target size (e.g. 512)
4. Output as JSON

"""
import argparse
import pandas as pd
import ast
import cv2
import json
import os
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    CF_DIR,
    DATASET_DIR,
    GROUPED_METADATA_PATH,
    PROCESSED_METADATA_PATH,
    PROCESSED_METADATA_512_PATH,
)

def parse_serialized_list(value):
    """Parse various malformed serialization formats to actual list."""
    try:
        parsed = ast.literal_eval(value)
        
        if isinstance(parsed, list):
            # Flatten nested lists and extract strings
            result = []
            for item in parsed:
                if isinstance(item, list):
                    result.extend([str(x) for x in item])
                elif isinstance(item, str):
                    try:
                        inner = ast.literal_eval(item)
                        if isinstance(inner, list):
                            result.extend([str(x) for x in inner])
                        else:
                            result.append(str(inner))
                    except:
                        result.append(item)
                else:
                    result.append(str(item))
            return result
        
        return [str(parsed)]
    
    except:
        print("Exception while parsing serialized list:", value)
        return []

def parse_coordinates(value):
    """Parse string representation of list to actual list of ints."""
    try:
        def is_valid_coord(x):
            if x is None:
                return False
            if isinstance(x, str) and x.strip() == "":
                return False
            if isinstance(x, float) and math.isnan(x):
                return False
            return True

        if isinstance(value, list):
            return [int(float(x)) for x in value if is_valid_coord(x)]

        if isinstance(value, (int, float)):
            if isinstance(value, float) and math.isnan(value):
                return []
            return [int(value)]

        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [int(float(x)) for x in parsed if is_valid_coord(x)]

        if is_valid_coord(parsed):
            return [int(float(parsed))]

        return []

    except Exception:
        print("Exception while parsing coordinates:", value)
        return []

def validate_bbox_coordinates(xmin_list, ymin_list, xmax_list, ymax_list):
    """
    Validate and correct bounding boxes.
    - If coordinate lists have different lengths, mark as invalid and return empty lists.
    - If xmin > xmax or ymin > ymax, swap to correct order.
    - If xmin == xmax or ymin == ymax, drop the bbox as invalid.
    """
    lengths = {len(xmin_list), len(ymin_list), len(xmax_list), len(ymax_list)}
    if len(lengths) != 1:
        return [], [], [], []

    valid_xmin, valid_ymin, valid_xmax, valid_ymax = [], [], [], []

    for xmin, ymin, xmax, ymax in zip(xmin_list, ymin_list, xmax_list, ymax_list):
        if xmin > xmax:
            xmin, xmax = xmax, xmin # Swap to correct order
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        if xmin == xmax or ymin == ymax:
            continue

        valid_xmin.append(xmin)
        valid_ymin.append(ymin)
        valid_xmax.append(xmax)
        valid_ymax.append(ymax)

    return valid_xmin, valid_ymin, valid_xmax, valid_ymax

def get_image_dimensions(image_dir, image_id):
    """Get image dimensions (width, height), trying with/without .png extension."""
    try:
        path = os.path.join(image_dir, image_id)
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        h, w = im.shape[:2]
        return w, h
    except Exception:
        raise Exception(f"Error reading image {path}")

def resize_bbox_with_aspect_ratio(xmin_list, ymin_list, xmax_list, ymax_list, 
                                   orig_width, orig_height, target_size=512):
    """
    Resize bounding boxes using aspect-ratio-preserving method.
    Same transformation as image resizing: scale + pad.
    """
    # When no bbox
    if len(xmin_list) == 0 or len(ymin_list) == 0 or len(xmax_list) == 0 or len(ymax_list) == 0:
        return [], [], [], []

    # Calculate scale factor (same as image resizing)
    scale = target_size / max(orig_width, orig_height)
    new_w = int(orig_width * scale)
    new_h = int(orig_height * scale)
    
    # Calculate padding (same as image resizing)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    resized_xmin, resized_ymin, resized_xmax, resized_ymax = [], [], [], []
    
    for i in range(len(xmin_list)):
        xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]
        
        # Skip zero/invalid bboxes
        if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
            continue
        
        # Apply scale
        xmin_scaled = int(xmin * scale)
        ymin_scaled = int(ymin * scale)
        xmax_scaled = int(xmax * scale)
        ymax_scaled = int(ymax * scale)
        
        # Apply padding
        xmin_padded = xmin_scaled + pad_w
        ymin_padded = ymin_scaled + pad_h
        xmax_padded = xmax_scaled + pad_w
        ymax_padded = ymax_scaled + pad_h
        
        # Clamp to target size
        xmin_padded = max(0, min(xmin_padded, target_size))
        ymin_padded = max(0, min(ymin_padded, target_size))
        xmax_padded = max(0, min(xmax_padded, target_size))
        ymax_padded = max(0, min(ymax_padded, target_size))
        
        resized_xmin.append(xmin_padded)
        resized_ymin.append(ymin_padded)
        resized_xmax.append(xmax_padded)
        resized_ymax.append(ymax_padded)
    
    return resized_xmin, resized_ymin, resized_xmax, resized_ymax

def process_dataframe(df_grouped, label='birads', resize_bboxes=False, target_size=None, image_dir=None, cf_dir=None):
    """
    Process grouped dataframe and (optionally) resize bboxes.
    Return list of dicts suitable for JSON output.
    """
    output = []

    image_dir = image_dir or DATASET_DIR
    cf_dir = cf_dir or CF_DIR
    cf_image_ids = set()
    if os.path.isdir(cf_dir):
        cf_image_ids = {
            name for name in os.listdir(cf_dir)
            if name.lower().endswith((".png"))
        }
    
    # Drop unnecessary columns and rename "training" split to "train"
    if label == 'birads':
        df_grouped = df_grouped.drop(columns=['Mass', 'Suspicious_Calcification'], errors='ignore')
    elif label == 'mass':
        df_grouped = df_grouped.drop(columns=['Suspicious_Calcification'], errors='ignore')
    elif label == 'calcification':
        df_grouped = df_grouped.drop(columns=['Mass'], errors='ignore')
    df_grouped.loc[df_grouped['split'] == 'training', 'split'] = 'train'

    for _, row in df_grouped.iterrows():
        record = row.to_dict()
        
        # Remove fold for test split (no cross-validation with the test set)
        if record.get('split') == 'test':
            record.pop('fold', None)
        
        # Process finding_categories
        if 'finding_categories' in record:
            # Remove unnecessary whitespaces and quotes
            finding_cats = parse_serialized_list(record['finding_categories'])
            # Special case: "No Finding"
            if finding_cats == ["No Finding"]:
                record['finding_categories'] = "No Finding"
                # Remove finding-related keys for "No Finding" records
                record.pop('finding_birads', None)
                record.pop('resized_xmin', None)
                record.pop('resized_ymin', None)
                record.pop('resized_xmax', None)
                record.pop('resized_ymax', None)
            else:
                record['finding_categories'] = finding_cats
                
                # Process finding_birads (unless it's "No Finding")
                if 'finding_birads' in record:
                    record['finding_birads'] = parse_serialized_list(record['finding_birads'])
                
                # Process coordinates for bbox resizing
                xmin_list = parse_coordinates(record.get('resized_xmin', []))
                ymin_list = parse_coordinates(record.get('resized_ymin', []))
                xmax_list = parse_coordinates(record.get('resized_xmax', []))
                ymax_list = parse_coordinates(record.get('resized_ymax', []))
                
                # Validate/correct coordinates before resizing
                xmin_list, ymin_list, xmax_list, ymax_list = validate_bbox_coordinates(
                    xmin_list, ymin_list, xmax_list, ymax_list
                )

                # Resize with aspect ratio preservation if target_size provided
                if resize_bboxes and target_size is not None and len(xmin_list) > 0:
                    rec_orig_w, rec_orig_h = get_image_dimensions(image_dir, record['image_id'])
                    
                    res_xmin, res_ymin, res_xmax, res_ymax = resize_bbox_with_aspect_ratio(
                        xmin_list, ymin_list, xmax_list, ymax_list,
                        rec_orig_w, rec_orig_h, target_size
                    )
                    record['resized_xmin'] = res_xmin
                    record['resized_ymin'] = res_ymin
                    record['resized_xmax'] = res_xmax
                    record['resized_ymax'] = res_ymax
                else:
                    # No target size provided: keep validated coordinates as-is
                    record['resized_xmin'] = xmin_list
                    record['resized_ymin'] = ymin_list
                    record['resized_xmax'] = xmax_list
                    record['resized_ymax'] = ymax_list

            # Add derived labels
            if label == 'birads':
                record['healthy'] = 1 if record.get('breast_birads') == 'BI-RADS 1' else 0
                if record.get('finding_categories') != 'No Finding' and record['image_id'] in cf_image_ids:
                    record['has_cf'] = 1
                else:
                    record['has_cf'] = 0
            
            elif label == 'mass' or label == 'calcification':
                column = 'Mass' if label == 'mass' else 'Suspicious_Calcification'

                if record.get(column) == 1:
                    record['healthy'] = 0 # In case of mass/calcification, "healthy" means absence of that finding
                    if record.get('finding_categories') != 'No Finding' and record['image_id'] in cf_image_ids:
                        record['has_cf'] = 1
                    else:
                        record['has_cf'] = 0
                else:
                    record['healthy'] = 1
                    record['has_cf'] = 0
        
        output.append(record)
    
    return output

def main():
    parser = argparse.ArgumentParser(description="Process grouped mammogram metadata.")
    parser.add_argument("--input-csv", default=GROUPED_METADATA_PATH)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--label", choices=["birads", "mass", "calcification"], default="birads")
    parser.add_argument("--image-dir", default=DATASET_DIR)
    parser.add_argument("--cf-dir", default=CF_DIR)
    parser.add_argument("--resize-bboxes", action="store_true")
    parser.add_argument("--target-size", type=int, default=512)
    args = parser.parse_args()

    df_grouped = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df_grouped)} images from {args.input_csv}")
    data_list = process_dataframe(
        df_grouped,
        label=args.label,
        resize_bboxes=args.resize_bboxes,
        target_size=args.target_size,
        image_dir=args.image_dir,
        cf_dir=args.cf_dir,
    )

    if args.output_json:
        output_path = args.output_json
    elif args.resize_bboxes and args.label == "birads" and args.target_size == 512:
        output_path = PROCESSED_METADATA_512_PATH
    elif args.label == "birads":
        output_path = PROCESSED_METADATA_PATH
    else:
        output_path = os.path.join(
            os.path.dirname(PROCESSED_METADATA_PATH),
            f"processed_df_{args.label}{'_' + str(args.target_size) if args.resize_bboxes else ''}.json",
        )

    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=2)
    print(f"Saved processed metadata to {output_path}")
    print(f"Total records: {len(data_list)}")


if __name__ == "__main__":
    main()

