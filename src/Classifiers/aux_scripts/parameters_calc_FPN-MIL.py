import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from config import DATASET_DIR

def calculate_mean_std():
    """Calculate mean and standard deviation of the dataset incrementally."""
    dataset_path = Path(DATASET_DIR)
    
    # First pass: calculate mean
    pixel_sum = None
    pixel_count = 0
    num_images = 0
    
    print("First pass: calculating mean...")
    for img_file in dataset_path.glob('**/*.png'):
        img = np.array(Image.open(img_file)) / 255.0  # Normalize to [0, 1]
        if len(img.shape) == 2:  # Grayscale
            img = img[..., np.newaxis]  # Add channel dimension
        
        if pixel_sum is None:
            pixel_sum = np.zeros(img.shape[-1])
        
        pixel_sum += np.sum(img, axis=(0, 1))
        pixel_count += img.shape[0] * img.shape[1]
        num_images += 1
    
    mean = pixel_sum / pixel_count
    print(f"Mean: {mean}")
    
    # Second pass: calculate std
    squared_diff_sum = np.zeros(mean.shape)
    num_images = 0
    
    print("\nSecond pass: calculating standard deviation...")
    for img_file in dataset_path.glob('**/*.png'):
        img = np.array(Image.open(img_file)) / 255.0
        if len(img.shape) == 2:
            img = img[..., np.newaxis]
        
        squared_diff_sum += np.sum((img - mean) ** 2, axis=(0, 1))
        num_images += 1
    
    std = np.sqrt(squared_diff_sum / pixel_count)
    print(f"\nStd: {std}")
    
    return mean, std

def calculate_bce_weights(csv_path):
    """Calculate BCE weights for class imbalance from metadata CSV."""
    print("\n" + "="*60)
    print("Calculating BCE weights from metadata...")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    
    weights = {}
    
    # Calculate for Mass
    mass_positive = (df['Mass'] == 1).sum()
    mass_negative = (df['Mass'] == 0).sum()
    mass_weight = mass_negative / mass_positive if mass_positive > 0 else 0
    weights['Mass'] = mass_weight
    
    print(f"\nMass:")
    print(f"  Positive samples: {mass_positive}")
    print(f"  Negative samples: {mass_negative}")
    print(f"  BCE weight: {mass_weight:.6f}")
    
    # Calculate for Suspicious_Calcification
    calc_positive = (df['Suspicious_Calcification'] == 1).sum()
    calc_negative = (df['Suspicious_Calcification'] == 0).sum()
    calc_weight = calc_negative / calc_positive if calc_positive > 0 else 0
    weights['Suspicious_Calcification'] = calc_weight
    
    print(f"\nSuspicious_Calcification:")
    print(f"  Positive samples: {calc_positive}")
    print(f"  Negative samples: {calc_negative}")
    print(f"  BCE weight: {calc_weight:.6f}")
    
    # Calculate for Anomaly (Mass OR Suspicious_Calcification)
    anomaly_positive = ((df['Mass'] == 1) | (df['Suspicious_Calcification'] == 1)).sum()
    anomaly_negative = len(df) - anomaly_positive
    anomaly_weight = anomaly_negative / anomaly_positive if anomaly_positive > 0 else 0
    weights['Anomaly'] = anomaly_weight
    
    print(f"\nAnomaly (Mass OR Suspicious_Calcification):")
    print(f"  Positive samples: {anomaly_positive}")
    print(f"  Negative samples: {anomaly_negative}")
    print(f"  BCE weight: {anomaly_weight:.6f}")
    
    print("\n" + "="*60)
    
    return weights

if __name__ == "__main__":
    # Calculate mean and std
    calculate_mean_std()
    
    # Calculate BCE weights
    # Assuming the CSV is in the data/metadata folder relative to project root
    csv_path = Path(__file__).parent.parent.parent.parent / "data" / "metadata" / "grouped_df.csv"
    if csv_path.exists():
        calculate_bce_weights(csv_path)
    else:
        print(f"\nWarning: CSV file not found at {csv_path}")
        print("Skipping BCE weight calculation.")