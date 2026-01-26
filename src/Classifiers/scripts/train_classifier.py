import argparse
import json
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import DATASET_DIR, MODELS_ROOT, METADATA_ROOT, IMAGES_ROOT
from src.Classifiers.aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset
from src.Classifiers.aux_scripts.plots_convnext import plot_training_metrics
from src.Classifiers.aux_scripts import logger
from src.Classifiers.aux_scripts.utils import create_transforms, train_epoch, validate_epoch, resume_from_checkpoint, unfreeze_layers
from src.Classifiers.aux_scripts.ClassifierConvNeXt import ConvNeXtClassifier
from src.Classifiers.aux_scripts.ClassifierVisionTransformer import VisionTransformerClassifier


def create_optimizer(model, args, log):
    """Create optimizer with appropriate learning rates for all model parameters.
    
    This function creates the optimizer including ALL model parameters (regardless of requires_grad),
    so that when layers are unfrozen mid-training, they will automatically be included with the
    correct learning rates.
    
    Args:
        model: The neural network model
        args: Training arguments
        log: Logger instance
    
    Returns:
        optimizer: Configured AdamW optimizer
    """
    # Differential learning rates
    if args.use_differential_lr:
        backbone_params = []
        classifier_params = []
        
        # Include ALL parameters (not just trainable ones)
        for name, param in model.named_parameters():
            if 'classifier' in name or 'heads' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': args.lr}  # Full LR for classifier
        ], weight_decay=args.weight_decay)
        
        log.info(f"Using differential learning rates: backbone LR={args.lr * 0.1:.2e}, classifier LR={args.lr:.2e}")
    
    # Uniform learning rate for all parameters
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        log.info(f"Using uniform learning rate: {args.lr:.2e}")
    
    return optimizer


def main():
    args = create_argparser().parse_args()
    
    # Set experiment name based on model type if not explicitly provided
    if args.experiment_name == 'classifier_training':
        args.experiment_name = f"{args.model_type}_classification"
    
    output_dir = logger.Logger.configure(experiment_type=args.model_type)
    if args.debugging:
        level = logging.DEBUG
    else:
        level = logging.INFO
    log = logger.Logger(log_dir=output_dir, log_file='training.log', level=level)
    
    # Save arguments
    args_path = os.path.join(output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    log.info(f"Training {args.model_type} model")

    # Create transforms
    train_transform, val_transform = create_transforms(args.augmentation_type)
    log.info(f"Using augmentation type: {args.augmentation_type}")
    
    # Create datasets using the updated VinDrMammo_dataset class
    train_dataset = VinDrMammo_dataset(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        split="training",
        transform=train_transform,
        # Use new flag-based system
        training_category=args.training_category,  # Use argument for category filtering
        training_cf=args.use_counterfactuals,  # Use new counterfactuals flag
        counterfactuals_dir=args.counterfactual_dir
    )
    log.debug(f"{len(train_dataset)} samples loaded from the training dataset.")
    
    # Create validation dataset (without counterfactuals)
    val_dataset = VinDrMammo_dataset(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        split="validation",  # This will use backward compatibility mode
        transform=val_transform,
        use_counterfactuals=False,  # Use old parameter for validation (backward compatibility)
        counterfactuals_dir=args.counterfactual_dir,
    )
    log.debug(f"{len(val_dataset)} samples loaded from the validation dataset.")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    log.debug("Training DataLoader created.")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    log.debug("Validation DataLoader created.")
    
    # Create model
    if args.model_type == "convnext":
        model = ConvNeXtClassifier(
            num_classes=1,
            pretrained=args.pretrained
        ).to(device)
        log.info("ConvNeXt model created and moved to device")
    elif args.model_type == "vit":
        model = VisionTransformerClassifier(
            num_classes=1,
            pretrained=args.pretrained
        ).to(device)
        log.info("ViT model created and moved to device")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Freeze initial feature layers/encoder blocks
    if args.freeze_layers > 0:
        frozen_count = 0

        if args.model_type == "convnext":
            for param in model.convnext.features[:args.freeze_layers].parameters():
                param.requires_grad = False
                frozen_count += 1
        
        elif args.model_type == "vit":
            for i, block in enumerate(model.vit.encoder.layers[:args.freeze_layers]):
                for param in block.parameters():
                    param.requires_grad = False
                    frozen_count += 1
        
        log.info(f"Frozen first {args.freeze_layers} layers of the {args.model_type} feature extractor")
    else:
        log.info("No layers frozen - training all parameters from start")
    
    # Create optimizer with appropriate learning rates
    optimizer = create_optimizer(model, args, log)

    # Calculate class weights from training dataset
    class_dist = train_dataset.get_class_distribution()
    num_neg = class_dist.get(0, 0)  # Healthy cases
    num_pos = class_dist.get(1, 0)  # Anomalous cases
    log.info(f"Training set - Healthy: {num_neg}, Anomalous: {num_pos}")
    
    # Direct ratio class weighting: num_negative / num_positive
    if num_pos > 0:
        pos_weight = torch.tensor([num_neg / num_pos]).to(device)
        log.info(f"Positive weight (num_neg/num_pos): {pos_weight.item():.4f}")
    else:
        pos_weight = torch.tensor([1.0]).to(device)
        log.warn("Training may be ineffective without positive samples.")
        
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Binary Cross Entropy
    log.debug("Loss function created.")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    log.debug("Learning rate scheduler created.")
    
    # Initialization for training loop
    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        checkpoint_path = os.path.join(MODELS_ROOT, args.resume_from_checkpoint)
        start_epoch, best_val_acc, best_val_loss = resume_from_checkpoint(
            checkpoint_path, model, optimizer, device
        )
    
    # Early stopping parameters
    patience = args.patience
    patience_counter = 0
    log.info(f"Early stopping patience: {patience} epochs")
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'val_precision': [], 'val_recall': [],
        'learning_rate': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        log.info(f"\n{'='*50}")
        log.info(f"Epoch {epoch+1}/{args.epochs}")
        log.info(f"{'='*50}")

        # Gradual unfreezing
        unfreeze_layers(model, epoch, args.epochs)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        log.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        log.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        log.info(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        log.info(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")

        # Early stopping based on validation loss (more stable than accuracy)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(output_dir, 'best_model.pth'))
            
            log.info(f"✓ New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            log.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            log.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Create training metrics plots
    if len(history['train_loss']) > 0:  # Only plot if we have training data
        plot_training_metrics(history, output_dir, args.experiment_name)
    
    log.info(f"\n{'='*50}")
    log.info(f"Training completed successfully!")
    log.info(f"Best validation loss: {best_val_loss:.4f}")
    log.info(f"Model and logs saved to: {output_dir}")
    log.info(f"{'='*50}")


def create_argparser():
    defaults = dict(
        data_dir=DATASET_DIR,
        metadata_path=os.path.join(METADATA_ROOT, 'resized_df_counterfactuals.csv'),
        experiment_name='classifier_training',  # Will be set based on model_type
        batch_size=16,
        epochs=100,
        patience=15,  # Early stopping patience (number of epochs without improvement)
        lr=3e-4,
        use_differential_lr=True,  # Use different learning rates for backbone and classifier
        weight_decay=0.01,
        num_workers=4,
        pretrained=True,
        freeze_layers=6,  # Number of initial feature layers to freeze (0 = no freezing)
        augmentation_type='standard',  # 'none', 'standard'
        use_counterfactuals=False,
        training_category=None,  # New: 'healthy', 'anomalous', 'anomalous_with_findings', or None
        counterfactual_dir=None,
        resume_from_checkpoint=None,
        debugging=False
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=defaults['data_dir'])
    parser.add_argument('--metadata_path', type=str, default=defaults['metadata_path'])
    parser.add_argument('--model_type', type=str, choices=['convnext', 'vit'], default="convnext")
    parser.add_argument('--experiment_name', type=str, default=defaults['experiment_name'])
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'])
    parser.add_argument('--epochs', type=int, default=defaults['epochs'])
    parser.add_argument('--patience', type=int, default=defaults['patience'],
                       help='Early stopping patience (number of epochs without improvement)')
    parser.add_argument('--lr', type=float, default=defaults['lr'])
    parser.add_argument('--use_differential_lr', action='store_true', default=defaults['use_differential_lr'],
                       help='Use different learning rates for backbone (lr*0.1) and classifier (lr)')
    parser.add_argument('--weight_decay', type=float, default=defaults['weight_decay'])
    parser.add_argument('--num_workers', type=int, default=defaults['num_workers'])
    parser.add_argument('--pretrained', action='store_true', default=defaults['pretrained'])
    parser.add_argument('--freeze_layers', type=int, default=defaults['freeze_layers'],
                       help='Number of initial feature layers to freeze (0 = no freezing)')
    parser.add_argument('--augmentation_type', type=str, choices=['none', 'standard'], 
                       default=defaults['augmentation_type'])
    parser.add_argument('--use_counterfactuals', action='store_true', default=defaults['use_counterfactuals'])
    parser.add_argument('--training_category', type=str, 
                       choices=['healthy', 'anomalous', 'anomalous_with_findings'], 
                       default=defaults['training_category'],
                       help='Filter training data by category')
    parser.add_argument('--counterfactual_dir', type=str, default=defaults['counterfactual_dir'])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument('--debugging', action='store_true', default=defaults['debugging'])
    
    return parser


if __name__ == "__main__":
    main()