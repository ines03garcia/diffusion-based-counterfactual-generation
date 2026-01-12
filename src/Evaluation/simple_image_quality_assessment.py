import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SimpleImageQualityAssessment:
    """
    Simplified Image Quality Assessment tool based on traditional computer vision metrics.
    No heavy dependencies required.
    """
    
    def __init__(self, counterfactuals_folder: str):
        self.folder_path = Path(counterfactuals_folder)
    
    def assess_image_quality(self, image_path: str) -> dict:
        """Assess quality of a single image using traditional metrics"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Could not load image'}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness (Laplacian variance) - higher is sharper
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Contrast (RMS contrast)
            contrast = np.std(gray)
            
            # 3. Brightness
            brightness = np.mean(gray)
            
            # 4. Noise estimation
            kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
            noise = cv2.filter2D(gray, -1, kernel)
            noise_level = np.var(noise)
            
            # 5. Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 6. Dynamic range
            dynamic_range = np.max(gray) - np.min(gray)
            
            # Calculate composite quality score
            # Normalize metrics and combine them
            sharpness_norm = min(sharpness / 1000, 1.0)  # Normalize sharpness
            contrast_norm = min(contrast / 100, 1.0)     # Normalize contrast
            brightness_norm = 1.0 - abs(brightness - 127.5) / 127.5  # Optimal brightness around middle
            noise_norm = max(0, 1.0 - noise_level / 10000)  # Lower noise is better
            edge_norm = min(edge_density * 10, 1.0)      # Normalize edge density
            range_norm = dynamic_range / 255             # Normalize dynamic range
            
            # Weighted composite score
            composite_score = (
                sharpness_norm * (1/4) +
                contrast_norm * (2/16) +
                brightness_norm * (1/16) +
                noise_norm * (1/4) +
                edge_norm * (1/4) +
                range_norm * (1/16)
            )
            
            return {
                'file_name': os.path.basename(image_path),
                'file_path': image_path,
                'sharpness': sharpness,
                'contrast': contrast,
                'brightness': brightness,
                'noise_level': noise_level,
                'edge_density': edge_density,
                'dynamic_range': dynamic_range,
                'composite_score': composite_score,
                'width': img.shape[1],
                'height': img.shape[0]
            }
            
        except Exception as e:
            return {'file_name': os.path.basename(image_path), 'error': str(e)}
    
    def assess_folder(self):
        """Assess all images in the folder"""
        if not self.folder_path.exists():
            print(f"Folder {self.folder_path} does not exist!")
            return None
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file in self.folder_path.iterdir():
            if file.suffix.lower() in image_extensions:
                image_files.append(file)
        
        print(f"Found {len(image_files)} images")
        
        # Assess each image
        results = []
        for i, img_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
            result = self.assess_image_quality(str(img_path))
            results.append(result)
        
        return pd.DataFrame(results)
    
    def create_report(self, df: pd.DataFrame):
        """Create a simple quality report"""
        # Save results
        df.to_csv('image_quality_results.csv', index=False)
        
        # Print summary
        print("\n" + "="*50)
        print("IMAGE QUALITY ASSESSMENT SUMMARY")
        print("="*50)
        
        valid_scores = df['composite_score'].dropna()
        if len(valid_scores) > 0:
            print(f"Total images: {len(df)}")
            print(f"Average quality score: {valid_scores.mean():.3f}")
            print(f"Best quality score: {valid_scores.max():.3f}")
            print(f"Worst quality score: {valid_scores.min():.3f}")
            
            # Quality categories
            high_quality = len(valid_scores[valid_scores >= 0.7])
            medium_quality = len(valid_scores[(valid_scores >= 0.4) & (valid_scores < 0.7)])
            low_quality = len(valid_scores[valid_scores < 0.4])
            
            print(f"\nQuality Distribution:")
            print(f"  High quality (≥0.7): {high_quality} images")
            print(f"  Medium quality (0.4-0.7): {medium_quality} images")
            print(f"  Low quality (<0.4): {low_quality} images")
            
            # Top and bottom performers
            print(f"\nTop 3 Quality Images:")
            top_3 = df.nlargest(3, 'composite_score')[['file_name', 'composite_score']]
            for _, row in top_3.iterrows():
                print(f"  {row['file_name']}: {row['composite_score']:.3f}")
            
            print(f"\nBottom 3 Quality Images:")
            bottom_3 = df.nsmallest(3, 'composite_score')[['file_name', 'composite_score']]
            for _, row in bottom_3.iterrows():
                print(f"  {row['file_name']}: {row['composite_score']:.3f}")
        
        print("\nDetailed results saved to 'image_quality_results.csv'")

# Usage
if __name__ == "__main__":
    folder_path = "/projects/F202507605CPCAA0/inescgarcia/data/counterfactuals"
    assessor = SimpleImageQualityAssessment(folder_path)
    results = assessor.assess_folder()
    
    if results is not None:
        assessor.create_report(results)