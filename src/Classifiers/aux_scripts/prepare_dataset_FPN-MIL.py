import os
import shutil
import pandas as pd
from pathlib import Path
from config import METADATA_PATH, SOURCE_DIR, DEST_DIR

# Define paths
metadata_path = METADATA_PATH
source_dir = SOURCE_DIR
dest_base_dir = DEST_DIR

# Read metadata
df = pd.read_csv(metadata_path)

# Create destination base directory if it doesn't exist
Path(dest_base_dir).mkdir(parents=True, exist_ok=True)

# Loop through each row in the dataframe
for _, row in df.iterrows():
    patient_id = row['patient_id']
    image_id = row['image_id']
    
    # Source image path
    source_image = os.path.join(source_dir, image_id)
    
    # Destination directory and path
    dest_patient_dir = os.path.join(dest_base_dir, str(patient_id))
    Path(dest_patient_dir).mkdir(parents=True, exist_ok=True)
    dest_image = os.path.join(dest_patient_dir, image_id)
    
    # Copy image if source exists
    if os.path.exists(source_image):
        shutil.copy2(source_image, dest_image)
        print(f"Copied: {source_image} -> {dest_image}")
    else:
        print(f"Warning: Source image not found: {source_image}")

print("Dataset preparation completed!")