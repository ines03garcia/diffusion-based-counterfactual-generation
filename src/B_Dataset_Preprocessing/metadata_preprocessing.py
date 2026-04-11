"""
Preprocess metadata:
1. Validate bounding boxes coordinates
2. Resize to 512x512
3. Output as JSON

"""
import pandas as pd
import ast
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
    return [] # Other type

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

def process_dataframe(df_grouped, orig_width=1520, orig_height=912, target_size=512):
    """
    Resize bboxes in grouped_df from 1520×912 to 512x512.
    Return list of dicts suitable for JSON output.
    """
    output = []
    cf_dir = "data/images/repaint_results"
    cf_image_ids = set()
    if os.path.isdir(cf_dir):
        cf_image_ids = {
            name for name in os.listdir(cf_dir)
            if name.lower().endswith((".png"))
        }
    
    for idx, row in df_grouped.iterrows():
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
                
                # Resize with aspect ratio preservation
                res_xmin, res_ymin, res_xmax, res_ymax = resize_bbox_with_aspect_ratio(
                    xmin_list, ymin_list, xmax_list, ymax_list,
                    orig_width, orig_height, target_size
                )
                
                # Store as native lists
                record['resized_xmin'] = res_xmin
                record['resized_ymin'] = res_ymin
                record['resized_xmax'] = res_xmax
                record['resized_ymax'] = res_ymax

            # Add derived labels
            record['healthy'] = 1 if record.get('breast_birads') == 'BI-RADS 1' else 0
            if record.get('finding_categories') != 'No Finding' and record['image_id'] in cf_image_ids:
                record['has_cf'] = 1
            else:
                record['has_cf'] = 0
        
        output.append(record)
    
    return output

if __name__ == "__main__":
    # Load data
    df_grouped = pd.read_csv("data/metadata/grouped_df.csv")
    print(f"Loaded {len(df_grouped)} images from grouped_df.csv")
    
    # Drop unnecessary columns and rename "training" split to "train"
    df_grouped.drop(columns=['Mass', 'Suspicious_Calcification'], errors='ignore', inplace=True)
    df_grouped.loc[df_grouped['split'] == 'training', 'split'] = 'train'
    
    print(f"Resizing bboxes from 1520×912 to 512×512...")
    # Resize bounding boxes and prepare for JSON
    data_list = process_dataframe(df_grouped, orig_width=1520, orig_height=912, target_size=512)
    
    # Save output as JSON
    output_path = "data/metadata/resized_df_512.json"
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=2)
    print(f"Saved resized bboxes to {output_path}")
    print(f"Total records: {len(data_list)}")

