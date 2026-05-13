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
from src.E_Aux_Scripts.argument_parsers import create_train_argparser


def build_base_setup(args):
    augmentation_mode = "cf_aug" if args.use_counterfactuals or args.add_cf_batch else "baseline_aug"
    
    if args.cross_validation:
        mode = "cross-validation"
    elif args.multiple_seeds:
        mode = "multiple_seeds"
    elif args.single_seed and args.no_validation:
        mode = "single_seed_no_validation"
    elif args.single_seed:
        mode = "single_seed"
    else:
        mode = "unknown_mode"
        
    return "/".join([augmentation_mode, mode])


def parse_and_validate_args():
    parser = create_train_argparser()
    args = parser.parse_args()

    if args.cross_validation and args.multiple_seeds:
        parser.error("Use only one mode: --cross-validation OR --multiple_seeds.")
    elif args.multiple_seeds and args.single_seed:
        parser.error("Use only one mode: --multiple_seeds OR --single_seed.")
    if not args.cross_validation and not args.multiple_seeds and not args.single_seed:
        parser.error("Enable one mode: --cross-validation, --multiple_seeds, or --single_seed.")
    if args.pair_loss_weight > 0.0 and not args.add_cf_batch:
        parser.error("--pair_loss_weight requires --add_cf_batch because pair indices are built from in-batch CF pairing.")
    if args.pair_loss_weight < 0.0:
        parser.error("--pair_loss_weight must be >= 0.")

    # Multi-seed runs are training-only (no validation split).
    if args.multiple_seeds:
        args.no_validation = True

    return args


def save_cf_low_scores(model, scoring_dataset, criterion, device, output_path, batch_size, num_workers, seed, log):
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
        if args.add_cf_batch and args.pair_loss_weight > 0.0:
            log.info(
                f"Joint objective enabled: total_loss = classification_loss + {args.pair_loss_weight} * pair_{args.pair_loss_type}_loss"
            )
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
                    add_cf_batch=args.add_cf_batch, 
                    cf_dir=args.cf_dir,
                    transform=train_transform,
                    use_mixup=args.use_mixup,
                    mixup_alpha=args.mixup_alpha,
                    pair_loss_weight=args.pair_loss_weight,
                    pair_loss_type=args.pair_loss_type,
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

            if args.loss == 'low' and args.use_counterfactuals:
                scoring_dataset = VinDrMammo_dataset(
                    split="train",
                    label=None,
                    cf_dir=args.cf_dir,
                    cv_fold=fold if not args.no_validation else None,
                    data_dir=args.data_dir,
                    metadata_path=args.metadata_path,
                    transform=val_transform,
                )
                low_scores_path = os.path.join(fold_results_dir, 'low_scores.csv')
                save_cf_low_scores(
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
                plot_training_metrics(history, fold_results_dir, validation=(not args.no_validation), use_mixup=args.use_mixup)

            log.info(f"\n{'='*15}Training completed successfully!{'='*15}")
            if not args.no_validation:
                log.info(f"Best validation loss: {best_val_loss:.4f}")
            else:
                log.info("No validation was used; training-only history was saved.")
            log.info(f"Model and logs saved to: {fold_results_dir}")




if __name__ == "__main__":
    main()