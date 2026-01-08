#!/usr/bin/env python3
"""
Script to create a validation split from training data and update the metadata CSV.

This script takes the training data and creates a validation split by randomly
selecting a percentage of training samples to be used for validation.
The original metadata CSV is updated with a new 'validation' split.
"""

import pandas as pd
import numpy as np
import argparse
import os


def create_validation_split(metadata_path, validation_split=0.2, random_seed=42, output_path=None):
    """
    Create a validation split from training data and update the metadata CSV.
    
    Args:
        metadata_path: Path to the original metadata CSV
        validation_split: Fraction of training data to use for validation (default: 0.2)
        random_seed: Random seed for reproducible splits (default: 42)
        output_path: Path to save the updated CSV (if None, overwrites original)
    
    Returns:
        str: Path to the updated CSV file
    """
    print(f"Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Show original split distribution
    original_splits = df['split'].value_counts()
    print(f"\nOriginal split distribution:")
    for split, count in original_splits.items():
        print(f"  {split}: {count}")
    
    # Filter training data
    training_df = df[df['split'] == 'training'].copy()
    other_df = df[df['split'] != 'training'].copy()
    
    print(f"\nFound {len(training_df)} training samples")
    
    if len(training_df) == 0:
        raise ValueError("No training data found in the metadata!")
    
    # Create reproducible split
    np.random.seed(random_seed)
    training_indices = training_df.index.tolist()
    np.random.shuffle(training_indices)
    
    # Calculate validation size
    val_size = int(len(training_indices) * validation_split)
    
    # Split indices
    val_indices = training_indices[:val_size]
    train_indices = training_indices[val_size:]
    
    print(f"Creating validation split with {val_size} samples ({validation_split*100:.1f}%)")
    print(f"Remaining training samples: {len(train_indices)}")
    
    # Update the splits
    df.loc[val_indices, 'split'] = 'validation'
    # training indices remain as 'training'
    
    # Show new split distribution
    new_splits = df['split'].value_counts()
    print(f"\nNew split distribution:")
    for split, count in new_splits.items():
        print(f"  {split}: {count}")
    
    # Show class distribution for each split
    print(f"\nClass distribution by split:")
    for split_name in ['training', 'validation', 'test']:
        if split_name in df['split'].values:
            split_df = df[df['split'] == split_name]
            
            # Count by BI-RADS (default anomaly type)
            birads_1_count = len(split_df[split_df['breast_birads'] == 'BI-RADS 1'])
            birads_other_count = len(split_df[split_df['breast_birads'] != 'BI-RADS 1'])
            
            print(f"  {split_name}:")
            print(f"    BI-RADS 1 (Healthy): {birads_1_count}")
            print(f"    BI-RADS != 1 (Anomalous): {birads_other_count}")
            
            # Count by Mass
            mass_0_count = len(split_df[split_df['Mass'] == 0])
            mass_1_count = len(split_df[split_df['Mass'] == 1])
            print(f"    Mass 0 (Healthy): {mass_0_count}")
            print(f"    Mass 1 (Anomalous): {mass_1_count}")
            
            # Count by Calcification
            calc_0_count = len(split_df[split_df['Suspicious_Calcification'] == 0])
            calc_1_count = len(split_df[split_df['Suspicious_Calcification'] == 1])
            print(f"    Suspicious_Calcification 0 (Healthy): {calc_0_count}")
            print(f"    Suspicious_Calcification 1 (Anomalous): {calc_1_count}")
    
    # Save the updated CSV
    if output_path is None:
        output_path = metadata_path
    
    df.to_csv(output_path, index=False)
    print(f"\nUpdated metadata saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create validation split from training data')
    parser.add_argument('--metadata_path', type=str, required=True,
                       help='Path to the metadata CSV file')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Fraction of training data to use for validation (default: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save updated CSV (default: overwrite original)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be done without actually modifying files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print(f"Would create validation split from: {args.metadata_path}")
        print(f"Validation split: {args.validation_split}")
        print(f"Random seed: {args.random_seed}")
        print(f"Output path: {args.output_path or args.metadata_path}")
        
        # Load and show current distribution
        df = pd.read_csv(args.metadata_path)
        splits = df['split'].value_counts()
        print(f"\nCurrent split distribution:")
        for split, count in splits.items():
            print(f"  {split}: {count}")
        
        training_count = len(df[df['split'] == 'training'])
        val_size = int(training_count * args.validation_split)
        print(f"\nWould create {val_size} validation samples from {training_count} training samples")
        return
    
    create_validation_split(
        metadata_path=args.metadata_path,
        validation_split=args.validation_split,
        random_seed=args.random_seed,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()