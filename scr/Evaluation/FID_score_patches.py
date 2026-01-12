import pandas as pd
import numpy as np
from PIL import Image
import os
import ast

metadata_csv= "../../data/metadata/resized_df_counterfactuals.csv"
images_base_dir = "../../data/VinDr-Mammo-Clip-resized-512"
counterfactual_dir = "../../data/counterfactuals_512"
patches_og = "../../data/patches_healthy_512"
patches_cf = "../../data/patches_counterfactuals_512"


metadata_df = pd.read_csv(metadata_csv)
healthy_df = metadata_df[metadata_df['breast_birads'] == 'BI-RADS 1']
anomalous_df = metadata_df[metadata_df['has_counterfactual'] == 1]

os.makedirs(patches_og, exist_ok=True)
os.makedirs(patches_cf, exist_ok=True)

import ast

def get_biggest_bbox_from_row(row):
    """
    Given a pandas row with columns 'resized_xmin', 'resized_ymin', 'resized_xmax', 'resized_ymax'
    (each containing a list of coordinates as a string), returns the biggest bbox as (xmin, ymin, xmax, ymax).
    """
    xmins = ast.literal_eval(row['resized_xmin'])
    ymins = ast.literal_eval(row['resized_ymin'])
    xmaxs = ast.literal_eval(row['resized_xmax'])
    ymaxs = ast.literal_eval(row['resized_ymax'])

    max_area = 0
    biggest_bbox = None

    for xmin, ymin, xmax, ymax in zip(xmins, ymins, xmaxs, ymaxs):
        print(f"Evaluating bbox: ({xmin}, {ymin}, {xmax}, {ymax})")
        
        if (xmin >= xmax) or (ymin >= ymax):
            print("Invalid bbox, skipping")
            continue
        
        area = max(0, xmax - xmin) * max(0, ymax - ymin)
        if area > max_area:
            max_area = area
            biggest_bbox = (xmin, ymin, xmax, ymax)

    if biggest_bbox is None:
        return None, None, None, None
    return biggest_bbox[0], biggest_bbox[1], biggest_bbox[2], biggest_bbox[3]


patches = []

def generate_patch_from_counterfactuals():
    for _, row in anomalous_df.iterrows():
        cf_path = os.path.join(counterfactual_dir, row['image_id'])
        patches_cf_path = os.path.join(patches_cf, row['image_id'])

        cf_img = np.array(Image.open(cf_path))

        resized_xmin, resized_ymin, resized_xmax, resized_ymax = get_biggest_bbox_from_row(row)
        if resized_xmin is None:
            print(f"No valid bbox found for image {row['image_id']}, skipping")
            continue

        print(f"Extracting patch for image {row['image_id']} at box ({resized_xmin}, {resized_ymin}, {resized_xmax}, {resized_ymax})")
        patches.append([resized_xmin, resized_ymin, resized_xmax, resized_ymax])
        
        # Patch of generated area corresponding to the fist lesion's bounding box
        """patch_cf = cf_img[resized_ymin:resized_ymax, resized_xmin:resized_xmax]

        # Resize for the FID computation
        img = Image.fromarray(patch_cf.astype(np.uint8))

        # Resize
        w, h = img.size
        target_size = 299
        scale = target_size / min(w, h)
        new_w, new_h = int(np.ceil(w * scale)), int(np.ceil(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to target_size x target_size
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        img = img.crop((left, top, right, bottom))
        img.save(patches_cf_path)"""


generate_patch_from_counterfactuals()


for _, row in healthy_df.iterrows():
    og_path = os.path.join(images_base_dir, row['image_id'])
    patches_og_path = os.path.join(patches_og, row['image_id'])

    og_img = np.array(Image.open(og_path))
    resized_xmin, resized_ymin, resized_xmax, resized_ymax = patches[0][0], patches[0][1], patches[0][2], patches[0][3]
    patches.pop(0)

    patch_og = og_img[resized_ymin:resized_ymax, resized_xmin:resized_xmax]

    # Resize for the FID computation
    img = Image.fromarray(patch_og.astype(np.uint8))

    # Resize
    w, h = img.size
    target_size = 299
    scale = target_size / min(w, h)
    new_w, new_h = int(np.ceil(w * scale)), int(np.ceil(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to target_size x target_size
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    img = img.crop((left, top, right, bottom))
    img.save(patches_og_path)
