import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import convnext_base
import argparse
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from code.config import DATASET_DIR, MODELS_ROOT, METADATA_ROOT, IMAGES_ROOT
from code.Classifiers.aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset


class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ConvNeXtClassifier, self).__init__()
        
        if pretrained:
            # Load from local file instead of downloading
            weights_path = os.path.join(MODELS_ROOT, "convnext_base-6075fbad.pth")
            if os.path.exists(weights_path):
                self.convnext = convnext_base(weights=None)  # Create model without weights
                state_dict = torch.load(weights_path, map_location='cpu')
                self.convnext.load_state_dict(state_dict)
                print(f"Loaded pretrained weights from {weights_path}")
            else:
                raise ValueError(f"Local weights not found at {weights_path}")
        else:
            self.convnext = convnext_base(weights=None)
        
        # Replace the classifier head
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, num_classes)
    
    def forward(self, x):
        #return self.convnext(x).squeeze(-1)
        return self.convnext(x)


def create_transforms(augmentation_type="standard"):
    """Create data transforms based on augmentation strategy"""
    
    if augmentation_type == "none":
        # No augmentation - only basic preprocessing
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization values
        ])
    else:
        # Standard data augmentation (and possibly with later on counterfactuals)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization values 
        ])
    
    # Validation transform is always the same
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization values 
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    for batch in tqdm(dataloader, desc="Training"):
        images, labels, image_names = batch
        images, labels = images.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(targets, predictions)
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images, labels, image_names = batch
            images, labels = images.to(device), labels.to(device).float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float() 
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, precision, recall, f1, predictions, targets


def unfreeze_layers(model, epoch, total_epochs):
    """Gradually unfreeze layers during training"""
    if epoch == total_epochs // 4:  # Unfreeze after 25% of training
        print("Unfreezing all feature layers...")
        for param in model.convnext.features.parameters():
            param.requires_grad = True
    elif epoch == total_epochs // 2:  # Unfreeze everything after 50%
        print("Unfreezing all layers...")
        for param in model.convnext.parameters():
            param.requires_grad = True


def plot_training_metrics(history, exp_dir, experiment_name):
    """Create and save plots showing training metrics evolution"""
    plt.style.use('default')
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Metrics Evolution - {experiment_name}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: F1 Score
    axes[0, 2].plot(epochs, history['val_f1'], 'g-', label='Validation F1', linewidth=2)
    axes[0, 2].set_title('Validation F1 Score', fontweight='bold')
    axes[0, 2].set_xlabel('Epochs')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    
    # Plot 4: Precision and Recall (if available)
    if 'val_precision' in history and 'val_recall' in history:
        axes[1, 0].plot(epochs, history['val_precision'], 'm-', label='Validation Precision', linewidth=2)
        axes[1, 0].plot(epochs, history['val_recall'], 'c-', label='Validation Recall', linewidth=2)
        axes[1, 0].set_title('Validation Precision and Recall', fontweight='bold')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
    else:
        axes[1, 0].axis('off')
    
    # Plot 5: Learning Rate (if available)
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], 'orange', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].axis('off')
    
    # Plot 6: Combined Overview
    ax_combined = axes[1, 2]
    ax2 = ax_combined.twinx()
    
    # Plot accuracy on left y-axis
    line1 = ax_combined.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax_combined.set_xlabel('Epochs')
    ax_combined.set_ylabel('Accuracy', color='r')
    ax_combined.tick_params(axis='y', labelcolor='r')
    ax_combined.set_ylim(0, 1)
    
    # Plot loss on right y-axis
    line2 = ax2.plot(epochs, history['val_loss'], 'b-', label='Val Loss', linewidth=2)
    ax2.set_ylabel('Loss', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax_combined.set_title('Validation Accuracy vs Loss', fontweight='bold')
    ax_combined.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_combined.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(exp_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to: {plot_path}")
    
    # Also create individual plots for each metric
    create_individual_plots(history, exp_dir, experiment_name)
    
    plt.close(fig)  # Close to free memory


def create_individual_plots(history, exp_dir, experiment_name):
    """Create individual plots for each metric"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Individual Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title(f'Loss Evolution - {experiment_name}', fontweight='bold', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'loss_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title(f'Accuracy Evolution - {experiment_name}', fontweight='bold', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'accuracy_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual F1 plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1'], 'g-', label='Validation F1 Score', linewidth=2)
    plt.title(f'F1 Score Evolution - {experiment_name}', fontweight='bold', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'f1_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual metric plots saved to: {exp_dir}")


def main():
    args = create_argparser().parse_args()
    
    # Add experimental configuration to name
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    exp_config = f"aug_{args.augmentation_type}_{args.anomaly_type}"
    
    # Add training category to config
    if args.training_category:
        exp_config += f"_{args.training_category}"
    else:
        exp_config += "_all"
    
    # Add counterfactuals flag
    if args.use_counterfactuals:
        exp_config += "_with_cf"
    
    args.experiment_name = f"{args.experiment_name}_{exp_config}_job{job_id}"
    
    # Create experiment directory
    exp_dir = os.path.join('Classifiers_logs', args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save arguments
    args_path = os.path.join(exp_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create transforms
    train_transform, val_transform = create_transforms(args.augmentation_type)
    
    # Create datasets using the updated VinDrMammo_dataset class
    train_dataset = VinDrMammo_dataset(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        split="training",
        transform=train_transform,
        # Use new flag-based system
        training_category=args.training_category,  # Use argument for category filtering
        training_cf=args.use_counterfactuals,  # Use new counterfactuals flag
        counterfactuals_dir=args.counterfactual_dir,
        anomaly_type=args.anomaly_type  # Pass anomaly type for classification
    )
    
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

    # Print dataset information
    print("Training dataset info:")
    print(train_dataset.get_split_info())
    print("\nValidation dataset info:")
    print(val_dataset.get_split_info())
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = ConvNeXtClassifier(
        num_classes=1,
        pretrained=args.pretrained
    ).to(device)

    # Differential learning rates - freeze feature layers initially
    for param in model.convnext.features[:-2].parameters():
        param.requires_grad = False
    
    # Set up optimizer with different learning rates
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for backbone
        {'params': classifier_params, 'lr': args.lr}       # Higher LR for classifier
    ], weight_decay=args.weight_decay)
    

    # Calculate class weights from training dataset
    class_dist = train_dataset.get_class_distribution()
    num_neg = class_dist.get(0, 0)  # Healthy cases
    num_pos = class_dist.get(1, 0)  # Anomalous cases
    
    print(f"Training set - Healthy: {num_neg}, Anomalous: {num_pos}")
    
    # Direct ratio class weighting: num_negative / num_positive
    if num_pos > 0:
        pos_weight = torch.tensor([num_neg / num_pos]).to(device)
        print(f"\nPositive weight (num_neg/num_pos): {pos_weight.item():.4f}\n")
    else:
        pos_weight = torch.tensor([1.0]).to(device)
        
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Binary Cross Entropy
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    # Early stopping parameters
    patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'val_precision': [], 'val_recall': [],
        'learning_rate': []
    }
    
    if args.resume_from_checkpoint:
        checkpoint_path = os.path.join(MODEL_ROOT, args.resume_from_checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Resume from next epoch
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['val_acc']
            
            print(f"Resumed from epoch {checkpoint['epoch']}, best val acc: {best_val_acc:.4f}")
    
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
            }, os.path.join(exp_dir, 'best_model.pth'))
            
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save training history
    with open(os.path.join(exp_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Create training metrics plots
    if len(history['train_loss']) > 0:  # Only plot if we have training data
        plot_training_metrics(history, exp_dir, args.experiment_name)
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")


def create_argparser():
    defaults = dict(
        data_dir=DATASET_DIR,
        metadata_path=os.path.join(METADATA_ROOT, 'resized_df_counterfactuals.csv'),
        experiment_name='convnext_classification',
        batch_size=16,
        epochs=100,
        lr=3e-4,
        weight_decay=0.01,
        num_workers=4,
        pretrained=True,
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
    parser.add_argument('--lr', type=float, default=defaults['lr'])
    parser.add_argument('--weight_decay', type=float, default=defaults['weight_decay'])
    parser.add_argument('--num_workers', type=int, default=defaults['num_workers'])
    parser.add_argument('--pretrained', action='store_true', default=defaults['pretrained'])
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