import logging
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import DATASET_DIR, METADATA_ROOT
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


def test_model(model, dataloader, device):
    """Test the model on the test set"""
    model.eval()
    predictions = []
    probabilities = []
    targets = []
    image_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            images, labels, names = batch
            images, labels = images.to(device), labels.to(device).float()
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            image_names.extend(names)
    
    return np.array(predictions), np.array(probabilities), np.array(targets), image_names


def calculate_metrics(predictions, probabilities, targets):
    """Calculate comprehensive metrics"""
    # Basic metrics
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='binary', zero_division=0)
    recall = recall_score(targets, predictions, average='binary', zero_division=0)
    f1 = f1_score(targets, predictions, average='binary', zero_division=0)
    
    # ROC AUC
    try:
        auc = roc_auc_score(targets, probabilities)
    except ValueError:
        auc = np.nan
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)
    
    # Specificity and Sensitivity
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        save_path = os.path.abspath(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")
        print(f"File exists: {os.path.exists(save_path)}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
        plt.close()  # Ensure figure is closed even if save fails


def plot_roc_curve(targets, probabilities, save_path):
    """Plot and save ROC curve"""
    from sklearn.metrics import roc_curve
    
    try:
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.abspath(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to: {save_path}")
        print(f"File exists: {os.path.exists(save_path)}")
        
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")
        plt.close()  # Ensure figure is closed even if save fails


def plot_probability_distribution(probabilities, targets, save_path):
    """Plot probability distribution for each class"""
    try:
        plt.figure(figsize=(10, 6))
        
        # Separate probabilities by true class
        healthy_probs = probabilities[targets == 0]
        anomalous_probs = probabilities[targets == 1]
        
        plt.hist(healthy_probs, bins=50, alpha=0.7, label='Healthy (True Label)', color='blue', density=True)
        plt.hist(anomalous_probs, bins=50, alpha=0.7, label='Anomalous (True Label)', color='red', density=True)
        
        plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution by True Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.abspath(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Probability distribution plot saved to: {save_path}")
        print(f"File exists: {os.path.exists(save_path)}")
    except Exception as e:
        print(f"Error saving probability distribution plot: {e}")
        plt.close()  # Ensure figure is closed even if save fails


def save_detailed_results(predictions, probabilities, targets, image_names, save_path):
    """Save detailed per-image results to CSV"""
    try:
        results_df = pd.DataFrame({
            'image_name': image_names,
            'true_label': targets,
            'predicted_label': predictions,
            'probability': probabilities,
            'correct': (predictions == targets).astype(int)
        })
        
        # Add interpretation columns
        results_df['true_class'] = results_df['true_label'].map({0: 'Healthy', 1: 'Anomalous'})
        results_df['predicted_class'] = results_df['predicted_label'].map({0: 'Healthy', 1: 'Anomalous'})
        
        # Add prediction confidence
        results_df['confidence'] = np.where(results_df['predicted_label'] == 1, 
                                           results_df['probability'], 
                                           1 - results_df['probability'])
        
        save_path = os.path.abspath(save_path)
        results_df.to_csv(save_path, index=False)
        print(f"Detailed results saved to: {save_path}")
        print(f"File exists: {os.path.exists(save_path)}")
        return results_df
    except Exception as e:
        print(f"Error saving detailed results: {e}")
        # Return empty dataframe as fallback
        return pd.DataFrame()


def log_metrics_summary(log, metrics):
    """Print a comprehensive metrics summary"""
    log.debug("\n" + "="*50)
    log.debug("           TEST SET RESULTS")
    log.debug("="*50)
    log.debug(f"Accuracy:     {metrics['accuracy']:.4f}")
    log.debug(f"Precision:    {metrics['precision']:.4f}")
    log.debug(f"Recall:       {metrics['recall']:.4f}")
    log.debug(f"F1-Score:     {metrics['f1_score']:.4f}")
    log.debug(f"Specificity:  {metrics['specificity']:.4f}")
    log.debug(f"AUC:          {metrics['auc']:.4f}")
    log.debug("\nConfusion Matrix:")
    log.debug(f"                 Predicted")
    log.debug(f"              Healthy  Anomalous")
    log.debug(f"Actual Healthy    {metrics['true_negatives']:4d}      {metrics['false_positives']:4d}")
    log.debug(f"    Anomalous     {metrics['false_negatives']:4d}      {metrics['true_positives']:4d}")
    log.debug("="*50)


def load_model(model_name, checkpoint_path, device, num_classes=1, log=None):
    """
    Args:
        model_name (str): Name of the model ('vit', 'convnext' or 'fpn-mil')
        checkpoint_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model on
        num_classes (int): Number of output classes (default: 1 for binary classification)
    
    Returns:
        torch.nn.Module: Loaded model
    """
    # Load checkpoint
    log.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model based on model name
    model_name = model_name.lower()
    if model_name == 'vit':
        log.info("Loading Vision Transformer model...")
        model = VisionTransformerClassifier(num_classes=num_classes, pretrained=False).to(device)
    elif model_name == 'convnext':
        log.info("Loading ConvNeXt model...")
        model = ConvNeXtClassifier(num_classes=num_classes, pretrained=False).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: 'vit', 'convnext'")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def save_metrics_json(log, metrics, output_dir, args):
    try:
        metrics_to_save = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        # Convert numpy types to native Python types for JSON serialization
        for key, value in metrics_to_save.items():
            if hasattr(value, 'item'):  # numpy scalar
                metrics_to_save[key] = value.item()
            elif isinstance(value, np.ndarray):
                metrics_to_save[key] = value.tolist()
        
        metrics_to_save['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        
        metrics_path = os.path.abspath(os.path.join(output_dir, 'test_metrics.json'))
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        log.debug(f"Test metrics saved to: {metrics_path}")
        log.debug(f"File exists: {os.path.exists(metrics_path)}")
    except Exception as e:
        log.error(f"Error saving test metrics: {e}")
        import traceback
        traceback.print_exc()

def save_test_arguments(log, args, output_dir):
    try:
        args_path = os.path.abspath(os.path.join(output_dir, 'test_args.json'))
        with open(args_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        log.info(f"Test arguments saved to: {args_path}")
        log.info(f"File exists: {os.path.exists(args_path)}")
    except Exception as e:
        log.error(f"Error saving test arguments: {e}")
        import traceback
        traceback.print_exc()

def log_saved_files_summary(log, output_dir):
    log.info(f"\nResults saved to: {output_dir}")
    log.info("Files created:")
    log.info("  - test_metrics.json: Overall performance metrics")
    log.info("  - detailed_results.csv: Per-image predictions and probabilities")
    log.info("  - confusion_matrix.png: Confusion matrix visualization")
    log.info("  - roc_curve.png: ROC curve")
    log.info("  - probability_distribution.png: Distribution of predicted probabilities")
    log.info("  - test_args.json: Test arguments used")


def log_class_summary(log, results_df):
    # Print class-wise performance
    log.info(f"\nClass-wise summary:")
    class_summary = results_df.groupby('true_class').agg({
        'correct': ['count', 'sum', 'mean'],
        'probability': ['mean', 'std']
    }).round(4)
    log.info(f"\n{class_summary}")


def main():
    args = create_argparser().parse_args()

    output_dir = logger.Logger.configure(experiment_type=f"classification_{args.model_type}")

    if args.debugging:
        level = logging.DEBUG
    else:
        level = logging.INFO
    log = logger.Logger(log_dir=output_dir, log_file='training.log', level=level)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_type, args.checkpoint_path, device, num_classes=1, log=log)
    
    # Create test dataset
    test_transform = create_test_transforms()
    
    test_dataset = VinDrMammo_dataset(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        split="test",
        transform=test_transform,
        use_counterfactuals=False  # Don't use counterfactuals for testing
    )
    
    log.debug(f"\nTest dataset info:")
    log.debug(test_dataset.get_split_info())
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Run testing
    log.info(f"\nTesting model on {len(test_dataset)} images...")
    predictions, probabilities, targets, image_names = test_model(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, probabilities, targets)
    
    # Log results
    if args.debugging:
        log_metrics_summary(log, metrics)
    
    # Save metrics to JSON
    save_metrics_json(log, metrics, output_dir, args)
    
    # Save detailed results
    log.info("\nSaving detailed results...")
    results_df = save_detailed_results(predictions, probabilities, targets, image_names, os.path.join(output_dir, 'detailed_results.csv'))
    
    # Create visualizations
    log.info("\nCreating visualizations...")
    class_names = ['Healthy', 'Anomalous']
    
    # Confusion matrix
    log.info("Plotting confusion matrix...")
    plot_confusion_matrix(metrics['confusion_matrix'], class_names,os.path.join(output_dir, 'confusion_matrix.png'))
    
    # ROC curve
    log.info("Plotting ROC curve...")
    plot_roc_curve(targets, probabilities,
                   os.path.join(output_dir, 'roc_curve.png'))
    
    # Probability distribution
    log.info("Plotting probability distribution...")
    plot_probability_distribution(probabilities, targets,
                                os.path.join(output_dir, 'probability_distribution.png'))
    
    # Save test arguments
    log.info("\nSaving test arguments...")
    save_test_arguments(log, args, output_dir)
    
    # Log saved files summary
    log_saved_files_summary(log, output_dir)

    # Log class-wise summary
    log_class_summary(log, results_df)


def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, required=True, choices=['convnext', 'vit'],
                       help='Type of model to load (convnext or vit)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default=DATASET_DIR,
                       help='Root directory containing the data')
    parser.add_argument('--metadata_path', type=str, 
                       default=os.path.join(METADATA_ROOT, 'resized_df_counterfactuals.csv'),
                       help='Path to the metadata CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--debugging', action='store_true', default=False,
                       help='Enable debugging mode with detailed logs')
    return parser


if __name__ == "__main__":
    main()

