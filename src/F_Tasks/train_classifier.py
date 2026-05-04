import argparse
import csv
import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
METADATA_ROOT = os.path.join(PROJECT_ROOT, "data/metadata")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data/images")

from src.C_Dataset_Handlers.VinDrMammo_dataset import VinDrMammo_dataset
from src.E_Aux_Scripts import logger
from src.E_Aux_Scripts.utils import set_seed, seed_worker
from src.E_Aux_Scripts.classifier_helpers import (
	build_model,
	create_transforms,
    create_optimizer,
    train_epoch,
    validate_epoch,
    resume_from_checkpoint,
    plot_training_metrics
)
from src.E_Aux_Scripts.LOW import LOWLoss


def build_base_setup(args):
    augmentation_mode = "cf_aug" if args.use_counterfactuals else "baseline_aug"
    
    if args.cross_validation:
        mode = "cross-validation"
    elif args.multiple_seeds:
        mode = "multiple_seeds"
    else:
        mode = "other"
        
    return "/".join([augmentation_mode, mode])


def parse_and_validate_args():
    parser = create_argparser()
    args = parser.parse_args()

    if args.cross_validation and args.multiple_seeds:
        parser.error("Use only one mode: --cross-validation OR --multiple_seeds.")
    if not args.cross_validation and not args.multiple_seeds:
        parser.error("Enable one mode: --cross-validation or --multiple_seeds.")

    # Multi-seed runs are training-only (no validation split).
    if args.multiple_seeds:
        args.no_validation = True

    return args


def save_low_scores(model, scoring_dataset, criterion, device, output_path, batch_size, num_workers, seed, log):
    scoring_loader = DataLoader(
        scoring_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    rows = []
    model.eval()

    with torch.enable_grad():
        sample_offset = 0
        for batch in scoring_loader:
            images, labels, image_ids = batch
            images = images.to(device)
            labels = labels.to(device)

            model.zero_grad(set_to_none=True)
            outputs = model(images)
            _, lossgrad, weights = criterion(outputs, labels.long(), return_details=True)
            sample_losses = criterion.loss(outputs, labels.long()).detach()

            for batch_index, (image_id, loss_value, grad_value, weight_value) in enumerate(
                zip(image_ids, sample_losses, lossgrad, weights)
            ):
                image_path = scoring_dataset.image_paths[sample_offset + batch_index]
                rows.append({
                    "rank": 0,
                    "image_id": image_id,
                    "label": int(labels[batch_index].item()),
                    "sample_loss": round(float(loss_value.item()), 4),
                    "loss_grad_norm": round(float(grad_value.item()), 4),
                    "low_score": round(float(weight_value.item()), 4),
                })

            sample_offset += len(image_ids)

    rows.sort(key=lambda row: row["low_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "rank",
                "image_id",
                "image_path",
                "label",
                "sample_loss",
                "loss_grad_norm",
                "low_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Saved LOW scores for {len(rows)} images to: {output_path}")


def main():
    args = parse_and_validate_args()
    if args.multiple_seeds:
        seeds = [args.seed + i for i in range(5)]
    else:
        seeds = [args.seed]

    base_setup = build_base_setup(args)
    
    # Create the main logger once for the entire run
    log = logger.Logger(experiment_type="Classifiers", sub_experiment_type="train", model_type=args.model_type, setup=base_setup)
    log.configure_root_logger()
    log.info(f"Logs will be saved to: {log.output_dir}")
    
    # Save arguments at the timestamp level
    args_path = os.path.join(log.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    for seed in seeds:
        set_seed(seed)
        
        # Create seed-specific subdirectory
        seed_dir = os.path.join(log.output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        log.info(f"Running with seed {seed} in directory: {seed_dir}")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device [{device}] to train model [{args.model_type}] on dataset [{args.dataset}] with [{'counterfactual augmentation' if args.use_counterfactuals else 'no counterfactuals'}] and [{'cross-validation' if args.cross_validation else 'no cross-validation'}].\n")

        # Create transforms
        train_transform, val_transform = create_transforms(
            augmentation_type=args.augmentation_type,
            model_type=args.model_type
        )

        log.info(f"Using augmentation type: {args.augmentation_type}")
        if args.use_mixup:
            log.info(f"MixUp augmentation enabled with alpha={args.mixup_alpha}")
        log.debug(f"Train transforms: {train_transform}")
        log.debug(f"Validation transforms: {val_transform}")

        # Log info
        results_dir = seed_dir
        log.info(f"Running training with seed {seed}. Results will be saved to: {results_dir}")
            
        if args.cross_validation:
            log.info("Performing cross-validation: val fold will start at 0, and the model will be trained and evaluated on each fold sequentially.")
            folds = 4
        else:
            folds = 1
        
        for fold in range(folds):
            fold_results_dir = results_dir
            if args.cross_validation:
                log.info(f"\n{'#'*15} Starting training for fold {fold} {'#'*15}")
                fold_results_dir = os.path.join(results_dir, f"fold_{fold}")
                os.makedirs(fold_results_dir, exist_ok=True)
            else:
                log.info(f"\n{'#'*15} Starting training without cross-validation {'#'*15}")

            # Create datasets using the updated VinDrMammo_dataset class
            train_dataset = VinDrMammo_dataset(
                split="train",
                label=args.training_category,
                cf_dir=args.cf_dir if args.use_counterfactuals else None,
                cv_fold=fold if not args.no_validation else None,
                data_dir=args.data_dir,
                metadata_path=args.metadata_path,
                transform=train_transform,
            )
            log.info(f"{len(train_dataset)} samples loaded from the training dataset.")
            
            if not args.no_validation:
                # Create validation dataset (without counterfactuals)
                val_dataset = VinDrMammo_dataset(
                    split="val",
                    label=args.training_category,
                    cf_dir=None, # Exclude cf from validation
                    cv_fold=fold,
                    data_dir=args.data_dir,
                    metadata_path=args.metadata_path,
                    transform=val_transform,
                )
                log.info(f"{len(val_dataset)} samples loaded from the validation dataset.")

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=lambda worker_id: seed_worker(seed, worker_id),
                generator=torch.Generator().manual_seed(seed)
            )

            if not args.no_validation:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True
                )
            
            log.info(f"Using batch size {args.batch_size} and {args.num_workers} workers.")

            # Create model
            model = build_model(args, device, experiment="train")
            
            # Freeze initial feature layers/encoder blocks
            if args.freeze_layers > 0:
                model.freeze_layers(args.freeze_layers)
                log.info(f"Frozen first {args.freeze_layers} layers of the {args.model_type} feature extractor")
            else:
                log.info("No frozen layers: training all parameters from start")
            
            # Create optimizer with appropriate learning rates
            optimizer = create_optimizer(model, args)

            if args.loss == 'bce':
                classes_distribution = train_dataset.get_class_distribution()
                if classes_distribution[1] > 0:
                    pos_weight = round(classes_distribution[0] / classes_distribution[1], 3)
                else:
                    pos_weight = 1.0
                    log.warning("NO POSITIVE SAMPLES IN THE TRAINING SET!")
                log.info(f"Class distribution in training set: {classes_distribution}, using pos_weight={pos_weight} for BCEWithLogitsLoss.")
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)).to(device)
            elif args.loss == 'low':
                criterion = LOWLoss(lamb=0.1).to(device)
            log.debug("Loss function created")

            # Learning rate scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.lr * 0.01
            )
            log.debug("Learning rate scheduler created.")

            # Initialization for training loop
            start_epoch = 0
            best_val_loss = float('inf')

            # Resume from checkpoint if specified
            if args.resume_from_checkpoint and fold == 0:  # Only resume for the first fold if doing cross-validation
                checkpoint_path = os.path.join(MODELS_ROOT, args.resume_from_checkpoint)
                start_epoch, best_val_loss = resume_from_checkpoint(
                    checkpoint_path, model, optimizer, device
                )

            if not args.no_validation:
                # Early stopping parameters
                patience = args.patience
                patience_counter = 0
                log.info(f"Early stopping patience: {patience} epochs")
            
            # Training history
            history = {
                'train_loss': [], 'train_acc': [],
                'train_f1': [],
                'learning_rate': []
            }
            if not args.no_validation:
                history['val_loss'] = []
                history['val_f1'] = []

            for epoch in range(start_epoch, args.epochs):
                epoch_start_time = time.time()
                log.info(f"\n{'='*15}Epoch {epoch+1}/{args.epochs}{'='*15}")

                # Gradual unfreezing
                if epoch == args.epochs // 4 or epoch == args.epochs // 2:  # Unfreeze after 25% and 50% of training
                    model.unfreeze_layers(epoch, args.epochs)

                # Train
                train_loss, train_acc, train_f1 = train_epoch(
                    model, train_loader, criterion, optimizer, device,
                    use_mixup=args.use_mixup,
                    mixup_alpha=args.mixup_alpha
                )
                
                if not args.no_validation:
                    # Validate
                    val_loss, val_f1, val_preds, val_targets = validate_epoch(
                        model, val_loader, criterion, device
                    )
                
                scheduler.step()

                # Record history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['train_f1'].append(train_f1)
                if not args.no_validation:
                    history['val_loss'].append(val_loss)
                    history['val_f1'].append(val_f1)
                    log.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
                    log.info(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
                history['learning_rate'].append(scheduler.get_last_lr()[0])
                log.info(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")

                if not args.no_validation:
                    # Early stopping based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save best model
                        save_path = os.path.join(fold_results_dir, 'best_model.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'args': vars(args)
                        }, save_path)
                        log.info(f"New best model saved. Val Loss: {val_loss:.4f}")
                    else:
                        patience_counter += 1
                        log.info(f"No improvement. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        log.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                
                epoch_end_time = time.time()
                log.info(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")

            if args.no_validation:
                # Save final model when no validation is used
                save_path = os.path.join(fold_results_dir, 'final_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args)
                }, save_path)
                log.info(f"Model trained for {args.epochs} epochs without validation and saved to: {save_path}")
            
            # Save training history
            with open(os.path.join(fold_results_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=2)

            if args.loss == 'low':
                scoring_dataset = VinDrMammo_dataset(
                    split="train",
                    label=args.training_category,
                    cf_dir=args.cf_dir if args.use_counterfactuals else None,
                    cv_fold=fold if not args.no_validation else None,
                    data_dir=args.data_dir,
                    metadata_path=args.metadata_path,
                    transform=val_transform,
                )
                low_scores_path = os.path.join(fold_results_dir, 'low_scores.csv')
                save_low_scores(
                    model,
                    scoring_dataset,
                    criterion,
                    device,
                    low_scores_path,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    seed=seed,
                    log=log,
                )

            # Create training metrics plots
            if len(history['train_loss']) > 0:
                if args.no_validation:
                    plot_training_metrics(history, fold_results_dir, validation=False)
                else:
                    plot_training_metrics(history, fold_results_dir, validation=True)

            log.info(f"\n{'='*15}Training completed successfully!{'='*15}")
            if not args.no_validation:
                log.info(f"Best validation loss: {best_val_loss:.4f}")
            else:
                log.info("No validation was used; training-only history was saved.")
            log.info(f"Model and logs saved to: {fold_results_dir}")


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['vindr'], default="vindr")
    parser.add_argument('--data_dir', type=str, default=os.path.join(IMAGES_ROOT, "VinDr-Mammo-Clip-CLAHE-512"))
    parser.add_argument('--metadata_path', type=str, default=os.path.join(METADATA_ROOT, "resized_df_512.json"))
    parser.add_argument('--training_category', type=str, choices=['all', 'healthy', 'anomalous', 'anomalous_with_findings'], default="all", help="Category of images to include in training")
    parser.add_argument('--cf_dir', type=str, default=os.path.join(IMAGES_ROOT, "repaint_results"))
    parser.add_argument('--use_counterfactuals', action='store_true', default=False, help='Whether to include counterfactual examples in training')
    validation_mode_group = parser.add_mutually_exclusive_group()
    validation_mode_group.add_argument('--cross-validation', action='store_true', default=False, help='Whether to perform cross-validation')
    validation_mode_group.add_argument('--no_validation', action='store_true', default=False, help='Train for the specified number of epochs without validation split and without early stopping.')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (number of epochs without improvement)')
    parser.add_argument('--use_differential_lr', action='store_true', default=True, help='Use different learning rates for backbone (lr*0.1) and classifier (lr)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--freeze_layers', type=int, default=6, help='Number of initial feature layers to freeze (0 = no freezing)')
    parser.add_argument('--augmentation_type', type=str, choices=['none', 'standard'], default="standard")
    parser.add_argument('--use_mixup', action='store_true', default=False, help='Whether to apply MixUp augmentation during training')
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Beta distribution parameter for MixUp (higher = more diverse mixing)')
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--multiple_seeds', action='store_true', default=False)
    parser.add_argument('--loss', type=str, choices=['bce', 'low'], default='bce', help="Loss function to use during training (default: Binary Cross-Entropy Loss with logits).")

    subparsers = parser.add_subparsers(dest='model_type', required=True)

    convnext_parser = subparsers.add_parser('convnext')
    convnext_parser.add_argument('--batch_size', type=int, default=16)
    convnext_parser.add_argument('--epochs', type=int, default=100)
    convnext_parser.add_argument('--lr', type=float, default=3e-4)

    vit_parser = subparsers.add_parser('vit')
    vit_parser.add_argument('--batch_size', type=int, default=16)
    vit_parser.add_argument('--epochs', type=int, default=100)
    vit_parser.add_argument('--lr', type=float, default=3e-4)

    mammo_clip_parser = subparsers.add_parser('mammo-clip')
    mammo_clip_parser.add_argument('--batch_size', type=int, default=8)
    mammo_clip_parser.add_argument('--epochs', type=int, default=30)
    mammo_clip_parser.add_argument('--lr', type=float, default=5e-5)
    mammo_clip_parser.add_argument("--clip_chk_pt_path", default=os.path.join(MODELS_ROOT, "b5-model-best-epoch-7.tar"), type=str, help="Path to Mammo-CLIP chkpt")
    mammo_clip_parser.add_argument("--arch", default="upmc_vindr_breast_clip_det_b5_period_n_lp", type=str)

    fpn_mil_parser = subparsers.add_parser('fpn-mil')
    fpn_mil_parser.add_argument('--n_class', type=int, default=1, help='Number of classes for classification (default: 1 for binary classification)')
    fpn_mil_parser.add_argument('--batch_size', type=int, default=8)
    fpn_mil_parser.add_argument('--epochs', type=int, default=30)
    fpn_mil_parser.add_argument('--lr', type=float, default=5e-5)
    fpn_mil_parser.add_argument("--clip_chk_pt_path", default=os.path.join(MODELS_ROOT, "b2-model-best-epoch-10.tar"), type=str, help="Path to Mammo-CLIP chkpt")
    fpn_mil_parser.add_argument("--arch", default="upmc_vindr_breast_clip_det_b2_period_n_lp", type=str)

    # FPN-MIL: Patch extraction
    fpn_mil_parser.add_argument("--img-size", default=[512, 512], type=int, nargs='*')
    fpn_mil_parser.add_argument("--feature_extraction", default='online', type=str)
    fpn_mil_parser.add_argument("--feat_dim", default=352, type=int)
    fpn_mil_parser.add_argument('--patching', action='store_true', default=True, help='Wether to perform patching on full-resolution images. If false, it will consider previously extracted patches that were saved in a directory (default: False)')
    fpn_mil_parser.add_argument('--source_image', type=str, default='patches', choices=['patches', 'full_image'])
    fpn_mil_parser.add_argument('--patch_size', type=int, default=128)
    fpn_mil_parser.add_argument('--overlap', type=float, nargs='*', default=[0.0])

    # FPN-MIL: MIL model parameters
    fpn_mil_parser.add_argument('--mil_type', default='pyramidal_mil', choices=[None, 'instance', 'embedding', 'pyramidal_mil'], type=str, help="MIL approach")
    fpn_mil_parser.add_argument('--pooling_type', default='gated-attention', choices=['max', 'mean', 'attention', 'gated-attention', 'pma'], type=str, help="MIL pooling operator")
    fpn_mil_parser.add_argument('--type_mil_encoder', default='mlp', choices=['mlp', 'sab', 'isab'], type=str, help="Type of MIL encoder.")
    fpn_mil_parser.add_argument('--fcl_attention_dim', type=int, default=128, metavar='N', help='parameter for attention (internal hidden units)')
    fpn_mil_parser.add_argument('--map_prob_func', type=str, default='softmax', choices=['softmax', 'sparsemax', 'entmax', 'alpha_entmax'])
    fpn_mil_parser.add_argument('--fcl_encoder_dim', type=int, default=256, help='parameter for set transformer (internal hidden units)')
    fpn_mil_parser.add_argument('--sab_num_heads', type=int, default=4, help='parameter for set transformer (number of self-attention heads in set attention blocks)')
    fpn_mil_parser.add_argument('--isab_num_heads', type=int, default=4, help='parameter for set transformer (number of self-attention heads in induced set attention blocks)')
    fpn_mil_parser.add_argument('--pma_num_heads', type=int, default=1, help='parameter for set transformer (number of self-attention heads in pooling by multihead attention)')
    fpn_mil_parser.add_argument('--num_encoder_blocks', type=int, default=2, help='parameter for set transformer (number of encoder layers)')
    fpn_mil_parser.add_argument('--trans_num_inds', type=int, default=20, help='parameter for set transformer (number of inducing points for the ISAB)')
    fpn_mil_parser.add_argument('--trans_layer_norm', type=bool, default=False)

    # FPN-MIL: Multi-scale MIL
    fpn_mil_parser.add_argument('--multi_scale_model', type=str, choices=['fpn', 'backbone_pyramid', 'msp'], default='fpn')
    fpn_mil_parser.add_argument('--scales', type=int, nargs='*', default=(16, 32, 128), help="List of scales to use for the multi-scale model.")
    fpn_mil_parser.add_argument('--fpn_dim', type=int, default=256)
    fpn_mil_parser.add_argument('--upsample_method', type=str, choices=['bilinear', 'nearest'], default='nearest')
    fpn_mil_parser.add_argument('--norm_fpn', type=bool, default=False)
    fpn_mil_parser.add_argument('--deep_supervision', action='store_true', default=True)
    fpn_mil_parser.add_argument('--type_scale_aggregator', type=str, choices=['concatenation', 'max_p', 'mean_p', 'attention', 'gated-attention'], default='gated-attention')

    # FPN-MIL: Regularization parameters
    fpn_mil_parser.add_argument('--drop_classhead', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the classification head (default: 0.)')
    fpn_mil_parser.add_argument('--drop_attention_pool', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the attention pooling mechanism (default: 0.)')
    fpn_mil_parser.add_argument('--drop_mha', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the attention pooling mechanism (default: 0.)')
    fpn_mil_parser.add_argument('--fcl_dropout', type=float, default=0.0)
    fpn_mil_parser.add_argument("--lamda", type=float, default=0.0,
                                help='lambda used for balancing cross-entropy loss and rank loss.')

    # FPN-MIL: Nested MIL
    fpn_mil_parser.add_argument('--nested_model', action='store_true', default=False)

    # debugging option not implemented yet

    return parser


if __name__ == "__main__":
    main()