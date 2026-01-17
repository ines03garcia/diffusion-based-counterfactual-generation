import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import argparse
import json
import os

from src.config import DATASET_DIR, MODELS_ROOT, METADATA_ROOT, IMAGES_ROOT
from src.Classifiers.aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset
from src.Classifiers.aux_scripts.plots_convnext import plot_training_metrics
from src.Classifiers.aux_scripts import logger
from src.Classifiers.aux_scripts.utils import create_transforms, check_internet_connection, train_epoch, validate_epoch, resume_from_checkpoint, unfreeze_layers

class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ConvNeXtClassifier, self).__init__()
        
        if pretrained:
            # Check if internet connection is available
            if check_internet_connection():
                try:
                    print("Internet connection detected. Downloading pretrained weights...")
                    self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
                    print("Successfully loaded pretrained weights from internet")
                except Exception as e:
                    print(f"Failed to download weights from internet: {e}")
                    print("Falling back to local weights...")
                    self._load_local_weights()
            else:
                print("No internet connection detected. Loading from local file...")
                self._load_local_weights()
        else:
            self.convnext = convnext_base(weights=None)
        
        # Replace the classifier head
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, num_classes)
    
    def _load_local_weights(self):
        """Load pretrained weights from local file"""
        weights_path = os.path.join(MODELS_ROOT, "convnext_base-6075fbad.pth")
        if os.path.exists(weights_path):
            self.convnext = convnext_base(weights=None)  # Create model without weights
            state_dict = torch.load(weights_path, map_location='cpu')
            self.convnext.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {weights_path}")
        else:
            raise ValueError(f"Local weights not found at {weights_path}")
    
    def forward(self, x):
        return self.convnext(x)


def main():
    args = create_argparser().parse_args()
    
    output_dir = logger.Logger.configure(experiment_type="convnext")
    
    # Save arguments
    args_path = os.path.join(output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.Logger.info(f"Using device: {device}")

    # Create transforms
    train_transform, val_transform = create_transforms(args.augmentation_type)
    logger.Logger.info(f"Using augmentation type: {args.augmentation_type}")
    
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
    logger.Logger.debug(f"{len(train_dataset)} samples loaded from the training dataset.")
    
    # Create validation dataset (without counterfactuals)
    val_dataset = VinDrMammo_dataset(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        split="validation",  # This will use backward compatibility mode
        transform=val_transform,
        use_counterfactuals=False,  # Use old parameter for validation (backward compatibility)
        counterfactuals_dir=args.counterfactual_dir,
        anomaly_type=args.anomaly_type  # Pass anomaly type for classification
    )
    logger.Logger.debug(f"{len(val_dataset)} samples loaded from the validation dataset.")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    logger.Logger.debug("Training DataLoader created.")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    logger.Logger.debug("Validation DataLoader created.")
    
    # Create model
    model = ConvNeXtClassifier(
        num_classes=1,
        pretrained=args.pretrained
    ).to(device)
    logger.Logger.debug("Model created.")

    # Freeze initial feature layers
    if args.freeze_layers > 0:
        for param in model.convnext.features[:args.freeze_layers].parameters():
            param.requires_grad = False
        logger.Logger.info(f"Frozen first {args.freeze_layers} layers of the feature extractor")
    else:
        logger.Logger.info("No layers frozen - training all parameters from start")
    
    # Differential learning rates
    if args.use_differential_lr:
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:  # Trainable parameters
                if 'classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': args.lr}  # Full LR for classifier
        ], weight_decay=args.weight_decay)
        
        logger.Logger.info(f"Using differential learning rates: backbone LR={args.lr * 0.1:.2e}, classifier LR={args.lr:.2e}")
    
    # Uniform learning rate for all parameters
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        logger.Logger.info(f"Using uniform learning rate: {args.lr:.2e}")
    

    # Calculate class weights from training dataset
    class_dist = train_dataset.get_class_distribution()
    num_neg = class_dist.get(0, 0)  # Healthy cases
    num_pos = class_dist.get(1, 0)  # Anomalous cases
    logger.Logger.info(f"Training set - Healthy: {num_neg}, Anomalous: {num_pos}")
    
    # Direct ratio class weighting: num_negative / num_positive
    if num_pos > 0:
        pos_weight = torch.tensor([num_neg / num_pos]).to(device)
        logger.Logger.info(f"Positive weight (num_neg/num_pos): {pos_weight.item():.4f}")
    else:
        pos_weight = torch.tensor([1.0]).to(device)
        logger.Logger.error("Training may be ineffective without positive samples.")
        
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Binary Cross Entropy
    logger.Logger.debug("Loss function created.")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    logger.Logger.debug("Learning rate scheduler created.")
    
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
    logger.Logger.info(f"Early stopping patience: {patience} epochs")
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'val_precision': [], 'val_recall': [],
        'learning_rate': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

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
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")

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
            
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Create training metrics plots
    if len(history['train_loss']) > 0:  # Only plot if we have training data
        plot_training_metrics(history, output_dir, args.experiment_name)
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")


def create_argparser():
    defaults = dict(
        data_dir=DATASET_DIR,
        metadata_path=os.path.join(METADATA_ROOT, 'resized_df_counterfactuals.csv'),
        experiment_name='convnext_classification',
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
        use_counterfactuals=False,  # Changed default to False for clearer control
        training_category=None,  # New: 'healthy', 'anomalous', 'anomalous_with_findings', or None
        counterfactual_dir=os.path.join(IMAGES_ROOT, 'counterfactuals_512'),
        resume_from_checkpoint=None,
        anomaly_type='birads'  # 'birads', 'mass', 'calcification'
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=defaults['data_dir'])
    parser.add_argument('--metadata_path', type=str, default=defaults['metadata_path'])
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
    parser.add_argument('--anomaly_type', type=str, choices=['birads', 'mass', 'calcification'],
                       default=defaults['anomaly_type'],
                       help='Type of anomaly classification to use')
    
    return parser


if __name__ == "__main__":
    main()