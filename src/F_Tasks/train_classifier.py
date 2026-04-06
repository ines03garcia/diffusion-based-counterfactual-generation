import argparse
import json
import logging
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
METADATA_ROOT = os.path.join(PROJECT_ROOT, "data/metadata")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data/images")

from src.Classifiers.aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset
from src.Classifiers.aux_scripts.plots_convnext import plot_training_metrics
from src.E_Aux_Scripts import logger
from src.E_Aux_Scripts.utils import create_transforms, train_epoch, validate_epoch, resume_from_checkpoint, unfreeze_layers
from src.D_Models.ClassifierConvNeXt import ConvNeXtClassifier
from src.D_Models.ClassifierVisionTransformer import VisionTransformerClassifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    set_seed(args.seed)
    
    log = logger.Logger(experiment_type="Classifiers", sub_experiment_type="train", model_type=args.model_type, setup=f"{'with_cf' if args.use_counterfactuals else 'no_cf'}")
    print(f"Logs will be saved to: {log.output_dir}")
    exit(1)

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['vindr'], default="vindr")
    parser.add_argument('--model_type', type=str, choices=['convnext', 'vit', 'mammo-clip', 'fpn-mil'], default="convnext")
    parser.add_argument('--data_dir', type=str, default=os.path.join(IMAGES_ROOT, "VinDr-Mammo-Clip-CLAHE-512"))
    parser.add_argument('--metadata_path', type=str, default=os.path.join(METADATA_ROOT, "resized_df_512.json"))
    parser.add_argument('--cf_dir', type=str, default=os.path.join(IMAGES_ROOT, "repaint_results"))
    parser.add_argument('--use_counterfactuals', action='store_true', default=False, help='Whether to include counterfactual examples in training')
    parser.add_argument('--cross-validation', action='store_true', default=True, help='Whether to perform cross-validation')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (number of epochs without improvement)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--use_differential_lr', action='store_true', default=True, help='Use different learning rates for backbone (lr*0.1) and classifier (lr)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--freeze_layers', type=int, default=6, help='Number of initial feature layers to freeze (0 = no freezing)')
    parser.add_argument('--augmentation_type', type=str, choices=['none', 'standard'], default="standard")
    parser.add_argument('--seed', type=int, default=0)
    
    # resume from checkpoint and debugging options not implemented yet

    return parser


if __name__ == "__main__":
    main()