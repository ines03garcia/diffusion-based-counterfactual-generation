import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from aux_scripts.config import DATA_ROOT, DATA_DIR, METADATA_ROOT, MODEL_ROOT
from aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset
from scripts.vision_transformer import VisionTransformerClassifier, create_transforms
from scripts.convNeXt import ConvNeXtClassifier


# -----------------------------
# Model loading
# -----------------------------
def model_load(checkpoint_path, model_type, device):
    """
    Load either ViT or ConvNeXt model based on model_type parameter.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        model_type (str): Either "vit" or "convnext" 
        device: PyTorch device (cuda/cpu)
    
    Returns:
        model: Loaded model ready for inference
    """
    if model_type.lower() == "vit":
        model = VisionTransformerClassifier(num_classes=1, pretrained=False)
    elif model_type.lower() == "convnext":
        model = ConvNeXtClassifier(num_classes=1, pretrained=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'vit' or 'convnext'")
    
    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    
    model.to(device)
    model.eval()
    print(f"Loaded {model_type.upper()} model from {checkpoint_path}")
    return model

def load_model(checkpoint_path, device):
    """Legacy function for backward compatibility - defaults to ViT"""
    return model_load(checkpoint_path, "vit", device)


# -----------------------------
# Prediction helper
# -----------------------------
def predict(model, image, device):
    with torch.no_grad():
        if image.ndim == 3:  # [C, H, W]
            image = image.unsqueeze(0)
        image = image.to(device)
        output = model(image)
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)
    return pred, prob  # 0 = healthy, 1 = anomalous, probability


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Classify counterfactual images using ViT or ConvNeXt models')
    parser.add_argument('--model_type', type=str, default='vit', choices=['vit', 'convnext'],
                        help='Type of model to use (vit or convnext)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (if not provided, uses default paths)')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_transform = create_transforms("none")

    # Determine checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Use default paths based on model type
        if args.model_type.lower() == "vit":
            checkpoint_path = os.path.join(MODEL_ROOT, "vit_555013_cf.pth") # ViT model path
        elif args.model_type.lower() == "convnext":
            checkpoint_path = os.path.join(MODEL_ROOT, "convnext_no_cf_555470.pth") # ConvNeXt model path
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Load model using the new function
    model = model_load(checkpoint_path, args.model_type, device)

    # Initialize dataset for loading test anomalous images with findings
    metadata_csv = os.path.join(METADATA_ROOT, "resized_df_counterfactuals.csv")
    counterfactuals_dir = os.path.join(DATA_DIR, "counterfactuals_512")
    
    # Use the new flag-based system to load only anomalous test cases with findings
    dataset = VinDrMammo_dataset(
        data_dir=DATA_ROOT,
        metadata_path=metadata_csv,
        split="test",
        testing_category="anomalous_with_findings",  # Only anomalous cases with counterfactuals
        testing_cf=False,  # Don't include counterfactuals in the dataset, we'll load them separately
        transform=val_transform,
        counterfactuals_dir=counterfactuals_dir
    )

    print(f"Found {len(dataset)} test anomalous images with findings for evaluation")
    print(f"Dataset configuration: {dataset.get_config_summary()}")

    if len(dataset) == 0:
        print("No test anomalous images with findings found. Exiting.")
        return

    results = []
    total_images_processed = 0
    originals_guessed_correctly = 0  # Original anomalous images correctly predicted as anomalous
    counterfactuals_guessed_correctly = 0  # Counterfactuals correctly predicted as healthy
    anomalous_switched_to_healthy = 0  # Originally anomalous images that switched to healthy in counterfactual
    healthy_switched_to_anomalous = 0  # Originally healthy images that switched to anomalous in counterfactual

    # Process each anomalous image and its counterfactual
    for i in tqdm(range(len(dataset)), desc="Evaluating anomalous images and counterfactuals"):
        try:
            # Get original image from dataset
            original_img, original_label_tensor, image_name = dataset[i]
            original_label = int(original_label_tensor.item())
            
            # Load counterfactual using the helper method
            _, cf_img, _ = dataset.get_image_and_counterfactual(image_name)
            
            # Predict on original image
            original_pred, original_prob = predict(model, original_img, device)
            
            # Predict on counterfactual image
            cf_pred, cf_prob = predict(model, cf_img, device) if cf_img is not None else (None, None)
            
            # Calculate logit shift (probability change)
            logit_shift = None
            if cf_prob is not None:
                logit_shift = cf_prob - original_prob  # Negative means probability decreased (good for anomalous->healthy)
            
            # Check if original was correctly classified as anomalous
            original_correct = (original_pred == original_label == 1)  # Should be anomalous (1)
            if original_correct:
                originals_guessed_correctly += 1
            
            # Check if counterfactual was correctly classified as healthy (expected label = 0)
            cf_correct = (cf_pred == 0) if cf_pred is not None else False
            if cf_correct:
                counterfactuals_guessed_correctly += 1
            
            # Check if image switched from anomalous to healthy (original pred was anomalous, cf pred is healthy)
            switched_to_healthy = (original_pred == 1 and cf_pred == 0) if cf_pred is not None else False
            if switched_to_healthy:
                anomalous_switched_to_healthy += 1
            
            # Check if image switched from healthy to anomalous (original pred was healthy, cf pred is anomalous)
            switched_to_anomalous = (original_pred == 0 and cf_pred == 1) if cf_pred is not None else False
            if switched_to_anomalous:
                healthy_switched_to_anomalous += 1
            
            # Get additional metadata for this image
            sample_info = dataset.get_sample_info(i)
            image_row = dataset.df[dataset.df['image_id'] == image_name].iloc[0]
            
            # Store results
            results.append({
                "image_id": image_name,
                "breast_birads": image_row['breast_birads'],
                "finding_categories": image_row['finding_categories'],
                "patient_id": image_row['patient_id'],
                "laterality": image_row['laterality'],
                "view": image_row['view'],
                "original_true_label": original_label,
                "original_pred": original_pred,
                "original_prob": original_prob,
                "original_correct": original_correct,
                "cf_pred": cf_pred,
                "cf_prob": cf_prob,
                "cf_correct": cf_correct,
                "logit_shift": logit_shift,
                "switched_to_healthy": switched_to_healthy,
                "switched_to_anomalous": switched_to_anomalous,
                "has_counterfactual": cf_img is not None
            })
            
            total_images_processed += 1
            
        except Exception as e:
            print(f"Error processing image at index {i}: {e}")
            continue

    # Calculate and display comprehensive metrics
    original_accuracy = 100 * originals_guessed_correctly / total_images_processed if total_images_processed > 0 else 0
    cf_accuracy = 100 * counterfactuals_guessed_correctly / total_images_processed if total_images_processed > 0 else 0
    switch_to_healthy_rate = 100 * anomalous_switched_to_healthy / total_images_processed if total_images_processed > 0 else 0
    switch_to_anomalous_rate = 100 * healthy_switched_to_anomalous / total_images_processed if total_images_processed > 0 else 0
    
    print(f"\n=== COUNTERFACTUAL EVALUATION RESULTS ===")
    print(f"1. Total images processed: {total_images_processed}")
    print(f"2. Original images guessed correctly (as anomalous): {originals_guessed_correctly}/{total_images_processed} ({original_accuracy:.2f}%)")
    print(f"3. Counterfactuals guessed correctly (as healthy): {counterfactuals_guessed_correctly}/{total_images_processed} ({cf_accuracy:.2f}%)")
    print(f"4. Images originally classified as anomalous that switched to healthy: {anomalous_switched_to_healthy}/{total_images_processed} ({switch_to_healthy_rate:.2f}%)")
    print(f"5. Images originally classified as healthy that switched to anomalous: {healthy_switched_to_anomalous}/{total_images_processed} ({switch_to_anomalous_rate:.2f}%)")
    
    # Additional analysis: flip rate among correctly classified originals
    correctly_classified_originals = [r for r in results if r['original_correct'] and r['has_counterfactual']]
    if correctly_classified_originals:
        successful_switches = sum(1 for r in correctly_classified_originals if r['switched_to_healthy'])
        success_switch_rate = 100 * successful_switches / len(correctly_classified_originals)
        print(f"\n=== ADDITIONAL ANALYSIS ===")
        print(f"Among correctly classified originals ({len(correctly_classified_originals)} images):")
        print(f"Successfully switched to healthy: {successful_switches}/{len(correctly_classified_originals)} ({success_switch_rate:.2f}%)")
    else:
        print(f"\nNo correctly classified original anomalous images found!")
    
    # Logit shift analysis
    print(f"\n=== LOGIT SHIFT ANALYSIS ===")
    valid_logit_shifts = [r['logit_shift'] for r in results if r['logit_shift'] is not None]
    
    if valid_logit_shifts:
        mean_logit_shift = np.mean(valid_logit_shifts)
        std_logit_shift = np.std(valid_logit_shifts)
        median_logit_shift = np.median(valid_logit_shifts)
        
        print(f"Overall logit shift statistics (n={len(valid_logit_shifts)}):")
        print(f"  Mean: {mean_logit_shift:.4f} ± {std_logit_shift:.4f}")
        print(f"  Median: {median_logit_shift:.4f}")
        print(f"  Range: [{min(valid_logit_shifts):.4f}, {max(valid_logit_shifts):.4f}]")
        
        # Analyze logit shift for correctly classified originals
        correct_original_shifts = [r['logit_shift'] for r in results if r['original_correct'] and r['logit_shift'] is not None]
        if correct_original_shifts:
            mean_correct_shift = np.mean(correct_original_shifts)
            std_correct_shift = np.std(correct_original_shifts)
            negative_shifts = sum(1 for shift in correct_original_shifts if shift < 0)
            negative_shift_rate = 100 * negative_shifts / len(correct_original_shifts)
            
            print(f"\nLogit shift for correctly classified originals (n={len(correct_original_shifts)}):")
            print(f"  Mean: {mean_correct_shift:.4f} ± {std_correct_shift:.4f}")
            print(f"  Negative shifts (probability decreased): {negative_shifts}/{len(correct_original_shifts)} ({negative_shift_rate:.1f}%)")
        
        # Analyze logit shift for successful switches (anomalous->healthy)
        successful_switch_shifts = [r['logit_shift'] for r in results if r['switched_to_healthy'] and r['logit_shift'] is not None]
        if successful_switch_shifts:
            mean_switch_shift = np.mean(successful_switch_shifts)
            std_switch_shift = np.std(successful_switch_shifts)
            
            print(f"\nLogit shift for successful switches (anomalous→healthy, n={len(successful_switch_shifts)}):")
            print(f"  Mean: {mean_switch_shift:.4f} ± {std_switch_shift:.4f}")
            print(f"  This shows how much the probability of being anomalous decreased")
        
        # Analyze cases where original was correctly classified but didn't switch
        correct_no_switch_shifts = [r['logit_shift'] for r in results if r['original_correct'] and not r['switched_to_healthy'] and r['logit_shift'] is not None]
        if correct_no_switch_shifts:
            mean_no_switch_shift = np.mean(correct_no_switch_shifts)
            std_no_switch_shift = np.std(correct_no_switch_shifts)
            
            print(f"\nLogit shift for correctly classified that didn't switch (n={len(correct_no_switch_shifts)}):")
            print(f"  Mean: {mean_no_switch_shift:.4f} ± {std_no_switch_shift:.4f}")
            print(f"  This shows why some images remained classified as anomalous")
    else:
        print("No valid logit shifts found!")
    
    # Additional breakdown by BI-RADS category
    print(f"\n=== Breakdown by BI-RADS Category ===")
    birads_stats = {}
    for result in results:
        birads = result['breast_birads']
        if birads not in birads_stats:
            birads_stats[birads] = {'total': 0, 'original_correct': 0, 'cf_correct': 0, 'switched_to_healthy': 0, 'switched_to_anomalous': 0}
        
        birads_stats[birads]['total'] += 1
        if result['original_correct']:
            birads_stats[birads]['original_correct'] += 1
        if result['cf_correct']:
            birads_stats[birads]['cf_correct'] += 1
        if result['switched_to_healthy']:
            birads_stats[birads]['switched_to_healthy'] += 1
        if result['switched_to_anomalous']:
            birads_stats[birads]['switched_to_anomalous'] += 1
    
    for birads, stats in birads_stats.items():
        orig_acc = 100 * stats['original_correct'] / stats['total'] if stats['total'] > 0 else 0
        cf_acc = 100 * stats['cf_correct'] / stats['total'] if stats['total'] > 0 else 0
        switch_to_healthy_rate = 100 * stats['switched_to_healthy'] / stats['total'] if stats['total'] > 0 else 0
        switch_to_anomalous_rate = 100 * stats['switched_to_anomalous'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{birads}: {stats['total']} cases | Original accuracy: {orig_acc:.1f}% | CF accuracy: {cf_acc:.1f}% | Anom→Healthy: {switch_to_healthy_rate:.1f}% | Healthy→Anom: {switch_to_anomalous_rate:.1f}%")

    # Save detailed results to CSV
    df = pd.DataFrame(results)
    output_file = "counterfactual_evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved detailed results to {output_file}")
    
    # Save summary statistics
    summary_stats = {
        'total_images_processed': total_images_processed,
        'originals_guessed_correctly': originals_guessed_correctly,
        'original_accuracy_percent': original_accuracy,
        'counterfactuals_guessed_correctly': counterfactuals_guessed_correctly,
        'cf_accuracy_percent': cf_accuracy,
        'anomalous_switched_to_healthy': anomalous_switched_to_healthy,
        'switch_to_healthy_rate_percent': switch_to_healthy_rate,
        'healthy_switched_to_anomalous': healthy_switched_to_anomalous,
        'switch_to_anomalous_rate_percent': switch_to_anomalous_rate
    }
    
    # Add logit shift statistics
    if valid_logit_shifts:
        summary_stats.update({
            'mean_logit_shift': mean_logit_shift,
            'std_logit_shift': std_logit_shift,
            'median_logit_shift': median_logit_shift,
            'min_logit_shift': min(valid_logit_shifts),
            'max_logit_shift': max(valid_logit_shifts),
            'num_valid_logit_shifts': len(valid_logit_shifts)
        })
        
        if correct_original_shifts:
            summary_stats.update({
                'mean_logit_shift_correct_originals': mean_correct_shift,
                'std_logit_shift_correct_originals': std_correct_shift,
                'negative_shifts_correct_originals': negative_shifts,
                'negative_shift_rate_correct_originals_percent': negative_shift_rate,
                'num_correct_original_shifts': len(correct_original_shifts)
            })
        
        if successful_switch_shifts:
            summary_stats.update({
                'mean_logit_shift_successful_switches': mean_switch_shift,
                'std_logit_shift_successful_switches': std_switch_shift,
                'num_successful_switch_shifts': len(successful_switch_shifts)
            })
        
        if correct_no_switch_shifts:
            summary_stats.update({
                'mean_logit_shift_no_switches': mean_no_switch_shift,
                'std_logit_shift_no_switches': std_no_switch_shift,
                'num_no_switch_shifts': len(correct_no_switch_shifts)
            })
    
    if correctly_classified_originals:
        summary_stats['switch_rate_for_correct_originals_percent'] = success_switch_rate
        summary_stats['correctly_classified_originals_count'] = len(correctly_classified_originals)
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = "counterfactual_evaluation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary statistics to {summary_file}")


if __name__ == "__main__":
    main()
