"""
Preprocess metadata:
1. Drop unnecessary columns and rename "training" split to "train"
2. Validate bounding boxes coordinates
3. (Optional) Resize to target size (e.g. 512)
4. Output as JSON

"""
import pandas as pd
import ast
import cv2
import json
import os

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
        if isinstance(value, list):
            return [int(x) for x in value if x]
        if isinstance(value, (int, float)):
            return [int(value)] if value else []
        
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [int(x) for x in parsed if x]
        return [int(parsed)]
    except:
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

def process_dataframe(df_grouped, label='birads', resize_bboxes=False, target_size=None, image_dir="data/images/VinDrMammo-CLIP", cf_dir="data/images/repaint_results"):
    """
    Process grouped dataframe and (optionally) resize bboxes.
    Return list of dicts suitable for JSON output.
    """
    output = []

    cf_dir = "data/images/repaint_results" # To set "has_cf" label
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

if __name__ == "__main__":
    # Load data
    df_grouped = pd.read_csv("data/metadata/grouped_df.csv")
    print(f"Loaded {len(df_grouped)} images from grouped_df.csv")
    
    label="calcification" # Options: "birads", "mass", "calcification"
    resize_bboxes = False
    target_size = 512

    # Drop unecessary columns, resize bounding boxes (optionally) and prepare JSON
    data_list = process_dataframe(df_grouped, label=label, resize_bboxes=resize_bboxes, target_size=target_size)
    
    # Save output as JSON
    output_path = f"data/metadata/processed_df_{label}_{target_size}.json" if resize_bboxes else f"data/metadata/processed_df_{label}.json"
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=2)
    print(f"Saved resized bboxes to {output_path}")
    print(f"Total records: {len(data_list)}")

