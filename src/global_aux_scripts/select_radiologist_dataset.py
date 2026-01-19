"""
Script to select 100 balanced mammograms from the counterfactuals folder.
Balances selection by BI-RADS and breast density.
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from src.config import IMAGES_ROOT, METADATA_ROOT


def main():
    # Define paths
    images_folder = Path(IMAGES_ROOT)/ "counterfactuals_updated"
    metadata_file = Path(METADATA_ROOT)/ "resized_df_counterfactuals.csv"
    output_folder = Path(IMAGES_ROOT)/ "counterfactuals_radiologist"
    output_csv = Path(METADATA_ROOT)/ "radiologist_dataset.csv"
    
    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv(metadata_file)
    
    # Filter for images with counterfactuals
    df_with_cf = df[df['has_counterfactual'] == 1].copy()
    print(f"Total images with counterfactuals in CSV: {len(df_with_cf)}")
    
    # Check which images actually exist in the folder
    print("Checking which images exist in folder...")
    existing_images = set(os.listdir(images_folder))
    df_existing = df_with_cf[df_with_cf['image_id'].isin(existing_images)].copy()
    print(f"Images with counterfactuals that exist in folder: {len(df_existing)}")
    
    if len(df_existing) < 100:
        print(f"ERROR: Not enough images available. Only found {len(df_existing)} images.")
        return
    
    # Show distribution
    print("\nDistribution by BI-RADS:")
    print(df_existing['breast_birads'].value_counts().sort_index())
    print("\nDistribution by Density:")
    print(df_existing['breast_density'].value_counts().sort_index())
    
    # Strategy: First take all images from densities with < 25, then divide remaining slots equally
    n_target = 100
    densities = ['DENSITY A', 'DENSITY B', 'DENSITY C', 'DENSITY D']
    threshold = 25
    
    print(f"\nStep 1: Identifying densities with < {threshold} images...")
    
    # Check availability for each density
    density_availability = {}
    limited_densities = []
    abundant_densities = []
    
    for density in densities:
        density_df = df_existing[df_existing['breast_density'] == density]
        count = len(density_df)
        density_availability[density] = count
        
        if count < threshold:
            limited_densities.append(density)
            print(f"  {density}: {count} images (< {threshold}) - will take all")
        else:
            abundant_densities.append(density)
            print(f"  {density}: {count} images (>= {threshold})")
    
    # Calculate how many images we'll take from limited densities
    images_from_limited = sum(density_availability[d] for d in limited_densities)
    remaining_to_select = n_target - images_from_limited
    
    print(f"\nStep 2: Taking all {images_from_limited} images from limited densities")
    print(f"Remaining to select: {remaining_to_select} from {len(abundant_densities)} abundant densities")
    
    # Calculate target per abundant density
    if len(abundant_densities) > 0:
        base_per_abundant = remaining_to_select // len(abundant_densities)
        remainder = remaining_to_select % len(abundant_densities)
        print(f"Base per abundant density: {base_per_abundant}, with {remainder} getting +1")
    
    selected_images = []
    
    # Select all images from limited densities
    for density in limited_densities:
        print(f"\n--- {density} (limited) ---")
        density_df = df_existing[df_existing['breast_density'] == density]
        print(f"Taking all {len(density_df)} images")
        selected_images.append(density_df)
    
    # Select from abundant densities
    for i, density in enumerate(abundant_densities):
        print(f"\n--- {density} (abundant) ---")
        density_df = df_existing[df_existing['breast_density'] == density]
        
        # Calculate target for this density
        target_for_density = base_per_abundant + (1 if i < remainder else 0)
        print(f"Target: {target_for_density} from {len(density_df)} available")
        
        # Get BI-RADS distribution for this density
        birads_counts = density_df['breast_birads'].value_counts().sort_index()
        print(f"BI-RADS distribution: {dict(birads_counts)}")
        
        # Balance by BI-RADS within this density
        n_birads = len(birads_counts)
        base_per_birads = target_for_density // n_birads
        birads_remainder = target_for_density % n_birads
        
        density_selected = []
        
        for j, (birads, count) in enumerate(birads_counts.items()):
            target = base_per_birads + (1 if j < birads_remainder else 0)
            n_select = min(target, count)
            
            if n_select > 0:
                birads_images = density_df[density_df['breast_birads'] == birads]
                selected = birads_images.sample(n=n_select, random_state=42)
                density_selected.append(selected)
                print(f"  {birads}: selected {n_select}/{count} (target: {target})")
        
        # Combine selections for this density
        density_combined = pd.concat(density_selected, ignore_index=True)
        
        # If we didn't reach target, add more from this density
        if len(density_combined) < target_for_density:
            needed = target_for_density - len(density_combined)
            selected_ids = set(density_combined['image_id'])
            remaining = density_df[~density_df['image_id'].isin(selected_ids)]
            if len(remaining) >= needed:
                extra = remaining.sample(n=needed, random_state=42)
                density_combined = pd.concat([density_combined, extra], ignore_index=True)
                print(f"  Added {needed} more to reach target")
        
        selected_images.append(density_combined)
        print(f"Total selected for {density}: {len(density_combined)}")
    
    # Combine all selected images
    df_selected = pd.concat(selected_images, ignore_index=True)
    
    print(f"\nFinal selection: {len(df_selected)} images")
    print("\nFinal distribution by BI-RADS:")
    print(df_selected['breast_birads'].value_counts().sort_index())
    print("\nFinal distribution by Density:")
    print(df_selected['breast_density'].value_counts().sort_index())
    
    # Add ID column (0-99)
    df_selected['radiologist_id'] = range(len(df_selected))
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying images to {output_folder}...")
    
    # Copy images
    radiologist_id = 0
    for image_id in df_selected['image_id']:
        src = images_folder / image_id
        dst = output_folder / f"{radiologist_id}_{image_id}"
        shutil.copy2(src, dst)
        radiologist_id += 1
    
    print(f"Copied {len(df_selected)} images")
    
    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_selected.to_csv(output_csv, index=False)
    print(f"\nSaved metadata to {output_csv}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
