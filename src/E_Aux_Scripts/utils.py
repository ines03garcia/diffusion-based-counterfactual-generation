import os
import torch
import logging
import torchvision.transforms as transforms
import socket
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """Check if internet connection is available"""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def create_transforms(
    augmentation_type="standard",
    model_type="convnext",
):
    """Create classifier transforms for ImageNet-based models and Mammo-CLIP."""
    if model_type == "mammo-clip": # Mammo-CLIP (UPMC) specific values
        normalize = transforms.Normalize(mean=[0.3089279, 0.3089279, 0.3089279], std=[0.25053555408335154, 0.25053555408335154, 0.25053555408335154])
    else: # ImageNet values
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
        )

    if augmentation_type == "none":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    for batch in tqdm(dataloader, desc="Training"):
        images, labels, _ = batch
        images, labels = images.to(device), labels.to(device).float().view(-1)
        
        optimizer.zero_grad()
        outputs = model(images).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(targets, predictions)
    epoch_f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images, labels, _ = batch
            images, labels = images.to(device), labels.to(device).float().view(-1)
            
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float() 
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    return epoch_loss, f1, predictions, targets


def unfreeze_layers(model, epoch, total_epochs):
    """Gradually unfreeze layers during training"""

    if epoch == total_epochs // 4:  # Unfreeze after 25% of training
        logger.info("Unfreezing all feature layers...")
        # Detect model type by checking attributes
        
        if hasattr(model, 'convnext'):
            # ConvNeXt model
            for param in model.convnext.features.parameters():
                param.requires_grad = True
        
        elif hasattr(model, 'vit'):
            # ViT model
            for param in model.vit.encoder.parameters():
                param.requires_grad = True
    
    elif epoch == total_epochs // 2:  # Unfreeze everything after 50%
        logger.info("Unfreezing all layers...")
        for param in model.parameters():
            param.requires_grad = True


def resume_from_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model and optimizer state from checkpoint"""

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')
    
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get checkpoint info
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    logger.info(f"Resumed from epoch {checkpoint['epoch']}, at path {checkpoint_path}, best val loss: {best_val_loss:.4f}")
    
    return start_epoch, best_val_loss

def plot_training_metrics(history, exp_dir):
    """Create and save plots showing training metrics evolution"""
    plt.style.use('default')
    
    # Keep only loss and F1 (train + validation) visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Metrics Evolution', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: F1
    if 'train_f1' in history:
        axes[1].plot(epochs, history['train_f1'], 'b-', label='Training F1', linewidth=2)
    axes[1].plot(epochs, history['val_f1'], 'g-', label='Validation F1', linewidth=2)
    axes[1].set_title('Training and Validation F1 Score', fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(exp_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to: {plot_path}")
    
    # Also create individual plots for each metric
    create_individual_plots(history, exp_dir)
    
    plt.close(fig)  # Close to free memory


def create_individual_plots(history, exp_dir):
    """Create individual plots for each metric"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Individual Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title(f'Loss Evolution', fontweight='bold', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'loss_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual F1 plot
    plt.figure(figsize=(10, 6))
    if 'train_f1' in history:
        plt.plot(epochs, history['train_f1'], 'b-', label='Training F1 Score', linewidth=2)
    plt.plot(epochs, history['val_f1'], 'g-', label='Validation F1 Score', linewidth=2)
    plt.title(f'F1 Score Evolution', fontweight='bold', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'f1_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Individual metric plots saved to: {exp_dir}")