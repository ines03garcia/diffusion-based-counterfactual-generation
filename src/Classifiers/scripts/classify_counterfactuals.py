import os
import logging
import torchvision.transforms as transforms
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

from src.config import DATASET_DIR, IMAGES_ROOT, METADATA_ROOT, MODELS_ROOT
from src.Classifiers.aux_scripts import logger
from src.Classifiers.aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset
from src.Classifiers.aux_scripts.ClassifierVisionTransformer import VisionTransformerClassifier
from src.Classifiers.aux_scripts.ClassifierConvNeXt import ConvNeXtClassifier

def create_test_transforms():
    """Create test transforms (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# -----------------------------
# Model loading
# -----------------------------
def load_model(checkpoint_path, model_type, device, num_classes=1, log=None):
    """
    Load either ViT or ConvNeXt model based on model_type parameter.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        model_type (str): Either "vit" or "convnext" 
        device: PyTorch device (cuda/cpu)
        num_classes (int): Number of output classes (default: 1 for binary classification)
        log: Logger instance
    
    Returns:
        model: Loaded model ready for inference
    """
    if log:
        log.info(f"Loading checkpoint from: {checkpoint_path}")
    
    if model_type.lower() == "vit":
        if log:
            log.info("Loading Vision Transformer model...")
        model = VisionTransformerClassifier(num_classes=num_classes, pretrained=False)
    elif model_type.lower() == "convnext":
        if log:
            log.info("Loading ConvNeXt model...")
        model = ConvNeXtClassifier(num_classes=num_classes, pretrained=False)
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
    return model


# -----------------------------
# Prediction helper
# -----------------------------
def predict(model, image, device):
    """Predict on a single image"""
    with torch.no_grad():
        if image.ndim == 3:  # [C, H, W]
            image = image.unsqueeze(0) # [1, C, H, W]
        image = image.to(device)
        output = model(image)
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)
    return pred, prob  # 0 = healthy or 1 = anomalous, probability


def evaluate_counterfactuals(model, dataset, device):
    """
    Evaluate counterfactuals by comparing original images with their counterfactual versions.
    
    Args:
        model: Trained classifier model
        dataset: Dataset containing anomalous images with counterfactuals
        device: PyTorch device
    
    Returns:
        results (list): List of dictionaries containing evaluation results for each image
    """
    results = []
    
    for i in tqdm(range(len(dataset)), desc="Evaluating counterfactuals"):
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
            
            # Check if counterfactual was correctly classified as healthy (expected label = 0)
            cf_correct = (cf_pred == 0) if cf_pred is not None else False
            
            # Check if image switched from anomalous to healthy (original pred was anomalous, cf pred is healthy)
            switched_to_healthy = (original_pred == 1 and cf_pred == 0) if cf_pred is not None else False
            
            # Check if image switched from healthy to anomalous (original pred was healthy, cf pred is anomalous)
            switched_to_anomalous = (original_pred == 0 and cf_pred == 1) if cf_pred is not None else False
            
            # Get additional metadata for this image
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
            
        except Exception as e:
            print(f"Error processing image at index {i}: {e}")
            continue
    
    return results


def calculate_counterfactual_metrics(results):
    """
    Calculate counterfactual evaluation metrics.
    
    Args:
        results (list): List of result dictionaries from evaluate_counterfactuals
    
    Returns:
        metrics (dict): Dictionary containing all calculated metrics
    """
    total_images = len(results)
    originals_correct = sum(1 for r in results if r['original_correct'])
    
    # Analysis: correctly classified originals
    correctly_classified_originals = [r for r in results if r['original_correct'] and r['has_counterfactual']]
    
    # CF Specificity: Among correctly classified originals, how many counterfactuals were correctly classified as healthy
    cf_correct_from_correct_originals = sum(1 for r in correctly_classified_originals if r['cf_correct'])
    cf_specificity_from_correct_originals = 100 * cf_correct_from_correct_originals / len(correctly_classified_originals) if correctly_classified_originals else 0
    
    # Opposite metric: Among wrongly classified originals (false negatives), how many counterfactuals were classified as anomalous
    wrongly_classified_originals = [r for r in results if not r['original_correct'] and r['original_true_label'] == 1 and r['has_counterfactual']]
    cf_anomalous_from_wrong_originals = sum(1 for r in wrongly_classified_originals if r['cf_pred'] == 1)
    cf_anomalous_from_wrong_originals_rate = 100 * cf_anomalous_from_wrong_originals / len(wrongly_classified_originals) if wrongly_classified_originals else 0
    
    # Correctly classified that didn't switch: originals correctly classified as anomalous, but CF remained anomalous
    correct_no_switch = [r for r in correctly_classified_originals if not r['switched_to_healthy']]
    correct_no_switch_count = len(correct_no_switch)
    correct_no_switch_rate = 100 * correct_no_switch_count / len(correctly_classified_originals) if correctly_classified_originals else 0
    
    # For cases that didn't switch, analyze if probability decreased (moved toward healthy)
    no_switch_with_decreased_prob = sum(1 for r in correct_no_switch if r['logit_shift'] is not None and r['logit_shift'] < 0)
    no_switch_decreased_prob_rate = 100 * no_switch_with_decreased_prob / correct_no_switch_count if correct_no_switch_count > 0 else 0
    
    metrics = {
        'total_images': total_images,
        'originals_correct': originals_correct,
        'correctly_classified_originals_count': len(correctly_classified_originals),
        'cf_correct_from_correct_originals': cf_correct_from_correct_originals,
        'cf_specificity_from_correct_originals': cf_specificity_from_correct_originals,
        'wrongly_classified_originals_count': len(wrongly_classified_originals),
        'cf_anomalous_from_wrong_originals': cf_anomalous_from_wrong_originals,
        'cf_anomalous_from_wrong_originals_rate': cf_anomalous_from_wrong_originals_rate,
        'correct_no_switch_count': correct_no_switch_count,
        'correct_no_switch_rate': correct_no_switch_rate,
        'no_switch_with_decreased_prob': no_switch_with_decreased_prob,
        'no_switch_decreased_prob_rate': no_switch_decreased_prob_rate
    }
    
    return metrics


def log_evaluation_summary(log, metrics):
    """Log evaluation summary"""
    log.info("\n" + "="*50)
    log.info("    COUNTERFACTUAL EVALUATION RESULTS")
    log.info("="*50)
    log.info(f"Total images processed: {metrics['total_images']}")
    log.info(f"Original images correctly classified (as anomalous): {metrics['originals_correct']}/{metrics['total_images']}")
    
    if metrics['correctly_classified_originals_count'] > 0:
        log.info("\n" + "="*50)
        log.info("  Among Correctly Classified Originals")
        log.info("="*50)
        log.info(f"Total: {metrics['correctly_classified_originals_count']} images")
        log.info(f"\n1. CF Specificity (CFs correctly classified as healthy):")
        log.info(f"   {metrics['cf_correct_from_correct_originals']}/{metrics['correctly_classified_originals_count']} ({metrics['cf_specificity_from_correct_originals']:.2f}%)")
        log.info(f"\n2. Did NOT switch (CFs remained anomalous):")
        log.info(f"   {metrics['correct_no_switch_count']}/{metrics['correctly_classified_originals_count']} ({metrics['correct_no_switch_rate']:.2f}%)")
        if metrics['correct_no_switch_count'] > 0:
            log.info(f"   → Of those, probability decreased (moved toward healthy):")
            log.info(f"     {metrics['no_switch_with_decreased_prob']}/{metrics['correct_no_switch_count']} ({metrics['no_switch_decreased_prob_rate']:.2f}%)")
    else:
        log.info("\nNo correctly classified original anomalous images found!")
    
    if metrics['wrongly_classified_originals_count'] > 0:
        log.info("\n" + "="*50)
        log.info("  Among Wrongly Classified Originals (False Negatives)")
        log.info("="*50)
        log.info(f"Total: {metrics['wrongly_classified_originals_count']} images")
        log.info(f"\nCounterfactuals classified as anomalous:")
        log.info(f"{metrics['cf_anomalous_from_wrong_originals']}/{metrics['wrongly_classified_originals_count']} ({metrics['cf_anomalous_from_wrong_originals_rate']:.2f}%)")
    else:
        log.info("\nNo wrongly classified original anomalous images found!")


def save_evaluation_results(results, metrics, output_dir):
    """
    Save evaluation results to CSV files.
    
    Args:
        results (list): List of result dictionaries
        metrics (dict): Calculated metrics
        output_dir (str): Directory to save results
    """
    # Save detailed results
    results_df = pd.DataFrame(results)
    detailed_path = os.path.join(output_dir, 'counterfactual_evaluation_results.csv')
    results_df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")
    
    # Prepare summary statistics for saving
    summary_stats = {
        'total_images_processed': metrics['total_images'],
        'originals_correctly_classified': metrics['originals_correct'],
        'correctly_classified_originals_count': metrics['correctly_classified_originals_count'],
        'cf_correct_from_correct_originals': metrics['cf_correct_from_correct_originals'],
        'cf_specificity_from_correct_originals_percent': metrics['cf_specificity_from_correct_originals'],
        'correct_no_switch_count': metrics['correct_no_switch_count'],
        'correct_no_switch_rate_percent': metrics['correct_no_switch_rate'],
        'no_switch_with_decreased_prob': metrics['no_switch_with_decreased_prob'],
        'no_switch_decreased_prob_rate_percent': metrics['no_switch_decreased_prob_rate'],
        'wrongly_classified_originals_count': metrics['wrongly_classified_originals_count'],
        'cf_anomalous_from_wrong_originals': metrics['cf_anomalous_from_wrong_originals'],
        'cf_anomalous_from_wrong_originals_rate_percent': metrics['cf_anomalous_from_wrong_originals_rate']
    }
    
    # Save summary statistics
    summary_df = pd.DataFrame([summary_stats])
    summary_path = os.path.join(output_dir, 'counterfactual_evaluation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")


# -----------------------------
# Arg parser
# -----------------------------
def create_argparser():
    defaults = dict(
        model_type="convnext",
        checkpoint_path=os.path.join(MODELS_ROOT, "ConvNeXt_no_cf.pth"),
        data_dir=DATASET_DIR,
        metadata_dir=os.path.join(METADATA_ROOT, "resized_df_counterfactuals.csv"),
        counterfactuals_dir=os.path.join(IMAGES_ROOT, "repaint_results"),
        batch_size=16,
    )
    parser = argparse.ArgumentParser(description="Classify counterfactual images")
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    parser.add_argument('--debugging', action='store_true', default=False,
                       help='Enable debugging mode with detailed logs')
    return parser


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    # Parse command line arguments
    args = create_argparser().parse_args()
    
    # Configure logging
    output_dir = logger.Logger.configure(experiment_type=f"counterfactual_evaluation_{args.model_type}")
    
    if args.debugging:
        level = logging.DEBUG
    else:
        level = logging.INFO
    log = logger.Logger(log_dir=output_dir, log_file='evaluation.log', level=level)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Create transforms
    val_transform = create_test_transforms()

    # Determine checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        raise ValueError("Error: --checkpoint_path must be provided.")
    
    # Load model
    model = load_model(checkpoint_path, args.model_type, device, num_classes=1, log=log)
    
    # Load dataset
    dataset = VinDrMammo_dataset(
        data_dir=DATASET_DIR,
        metadata_path=args.metadata_dir,
        split="test",
        testing_category="anomalous_with_findings",  # Only anomalous cases with counterfactuals
        testing_cf=False,  # Don't include counterfactuals in the dataset, we'll load them separately
        transform=val_transform,
        counterfactuals_dir=args.counterfactuals_dir
    )

    log.info(f"Found {len(dataset)} test anomalous images with findings for evaluation")
    log.info(f"Dataset configuration: {dataset.get_config_summary()}")

    if len(dataset) == 0:
        log.warning("No test anomalous images with findings found. Exiting.")
        return

    # Run evaluation
    log.info(f"\nEvaluating counterfactuals on {len(dataset)} images...")
    results = evaluate_counterfactuals(model, dataset, device)
    
    # Calculate metrics
    log.info("\nCalculating metrics...")
    metrics = calculate_counterfactual_metrics(results)
    
    # Log results
    log_evaluation_summary(log, metrics)
    
    # Save results
    log.info("\nSaving evaluation results...")
    save_evaluation_results(results, metrics, output_dir)
    
    log.info(f"\n{'='*50}")
    log.info(f"Results saved to: {output_dir}")
    log.info("Files created:")
    log.info("  - counterfactual_evaluation_results.csv: Detailed per-image results")
    log.info("  - counterfactual_evaluation_summary.csv: Summary statistics")
    log.info("  - evaluation.log: Evaluation log")
    log.info(f"{'='*50}")


if __name__ == "__main__":
    main()
