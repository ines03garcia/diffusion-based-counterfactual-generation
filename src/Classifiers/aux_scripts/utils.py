import os
import torch
import torchvision.transforms as transforms
import socket
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from src.Classifiers.aux_scripts import logger


def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """Check if internet connection is available"""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Validation transform is always the same
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        print("Unfreezing all layers...")
        for param in model.parameters():
            param.requires_grad = True


def resume_from_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model and optimizer state from checkpoint"""

    if not os.path.exists(checkpoint_path):
        logger.Logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 0, 0.0, float('inf')
    
    logger.Logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get checkpoint info
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint.get('val_acc', 0.0)
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    logger.Logger.info(f"Resumed from epoch {checkpoint['epoch']}, best val acc: {best_val_acc:.4f}, best val loss: {best_val_loss:.4f}")
    
    return start_epoch, best_val_acc, best_val_loss