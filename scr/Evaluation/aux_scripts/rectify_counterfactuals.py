from PIL import Image
import numpy as np
import pandas as pd
import os

metadata_csv= "../../data/metadata/resized_df_counterfactuals.csv"
images_base_dir = "../../data/VinDr-Mammo-Clip-resized-512"
counterfactual_dir = "../../data/counterfactuals"
masks_dir = "../../data/masks_512"
new_cf_dir = "../../data/rectified_counterfactuals"


metadata_df = pd.read_csv(metadata_csv)
metadata_df = metadata_df[metadata_df['has_counterfactual'] == 1]

mse = 0
os.makedirs(new_cf_dir, exist_ok=True)

for _, row in metadata_df.iterrows():
    img_path = os.path.join(images_base_dir, row['image_id'])
    cf_path = os.path.join(counterfactual_dir, row['image_id'])
    mask_path = os.path.join(masks_dir, row['image_id'])
    new_cf_path =  os.path.join(new_cf_dir, row['image_id'])

    original_img = np.array(Image.open(img_path))
    cf_img = np.array(Image.open(cf_path))
    mask_img = np.array(Image.open(mask_path))
    binary_mask = mask_img / 255.0

    masked_img = original_img * binary_mask
    new_cf = cf_img * (1 - binary_mask) + masked_img
    masked_cf = new_cf * binary_mask
    mse += np.mean((masked_img - masked_cf) ** 2)

    Image.fromarray(new_cf.astype(np.uint8)).save(new_cf_path)


print("Average MSE:", mse / len(metadata_df))