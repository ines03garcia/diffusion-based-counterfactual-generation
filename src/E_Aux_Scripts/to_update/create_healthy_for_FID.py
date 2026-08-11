import pandas as pd
import shutil
import os

metadata_csv= "../../../data/metadata/resized_df_counterfactuals.csv"
images_base_dir = "../../../data/images/VinDr-Mammo-Clip-CLAHE-512"
original_ds_for_FID = "../../../data/images/healthy_for_FID"

metadata_df = pd.read_csv(metadata_csv)
print(f"Loaded {len(metadata_df)} rows from metadata")

# Filter for 1767 healthy images (BI-RADS 1)
metadata_df = metadata_df[metadata_df['breast_birads'] == 'BI-RADS 1']
metadata_df = metadata_df.head(1767)

os.makedirs(original_ds_for_FID, exist_ok=True)

# And copy images to the new directory "healthy_for_FID"
for _, row in metadata_df.iterrows():
    img_path = os.path.join(images_base_dir, row['image_id'])
    original_img_name = os.path.join(original_ds_for_FID, row['image_id'])
    if os.path.exists(img_path):
        shutil.copy(img_path, original_img_name)

