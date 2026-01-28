import os
import shutil
import pandas as pd
from pathlib import Path
from config import METADATA_ROOT, IMAGES_ROOT, DATASET_DIR

# Define paths
metadata_path = os.path.join(METADATA_ROOT, "resized_df_has_counterfactual.csv")
source_dir = DATASET_DIR
counterfactuals_dir = os.path.join(IMAGES_ROOT, "repaint_results")
dest_base_dir = os.path.join(IMAGES_ROOT, "data_dir/img_dir")
dest_base_dir_cf = os.path.join(IMAGES_ROOT, "data_dir/img_dir_cf")

# Read metadata
df = pd.read_csv(metadata_path)

# Create destination base directories if they don't exist
Path(dest_base_dir).mkdir(parents=True, exist_ok=True)
Path(dest_base_dir_cf).mkdir(parents=True, exist_ok=True)

# Loop through each row in the dataframe
for _, row in df.iterrows():
    patient_id = row['patient_id']
    image_id = row['image_id']
    has_counterfactual = row.get('has_counterfactual', 0)
    
    # Source image path
    source_image = os.path.join(source_dir, image_id)
    
    # Destination directory and path for original dataset
    dest_patient_dir = os.path.join(dest_base_dir, str(patient_id))
    Path(dest_patient_dir).mkdir(parents=True, exist_ok=True)
    dest_image = os.path.join(dest_patient_dir, image_id)
    
    # Copy original image if source exists
    if os.path.exists(source_image):
        shutil.copy2(source_image, dest_image)
        print(f"Copied: {source_image} -> {dest_image}")
    else:
        print(f"Warning: Source image not found: {source_image}")
    
    # Prepare counterfactual dataset
    dest_patient_dir_cf = os.path.join(dest_base_dir_cf, str(patient_id))
    Path(dest_patient_dir_cf).mkdir(parents=True, exist_ok=True)
    dest_image_cf = os.path.join(dest_patient_dir_cf, image_id)
    
    # Always copy the original image to img_dir_cf
    if os.path.exists(source_image):
        shutil.copy2(source_image, dest_image_cf)
        print(f"Copied original to CF dir: {source_image} -> {dest_image_cf}")
    
    # If has_counterfactual == 1, also copy the counterfactual with modified filename
    if has_counterfactual == 1:
        cf_source_image = os.path.join(counterfactuals_dir, image_id)
        # Modify image_id to add _cf before extension
        image_name, image_ext = os.path.splitext(image_id)
        cf_image_id = f"{image_name}_cf{image_ext}"
        dest_cf_image = os.path.join(dest_patient_dir_cf, cf_image_id)
        
        if os.path.exists(cf_source_image):
            shutil.copy2(cf_source_image, dest_cf_image)
            print(f"Copied CF: {cf_source_image} -> {dest_cf_image}")
        else:
            print(f"Warning: Counterfactual image not found: {cf_source_image}")

print("Dataset preparation completed!")