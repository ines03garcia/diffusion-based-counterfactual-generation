import argparse
import csv
import json
import os
import sys
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torch.amp import GradScaler

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
    plot_training_metrics,
    LinearWarmupCosineAnnealingLR,
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

    # Multi-seed runs are training-only (no validation split).
    if args.multiple_seeds:
        args.no_validation = True

    return args


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def save_checkpoint(save_path, model, optimizer, epoch, args, val_loss=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, save_path)


def save_cf_low_scores(model, scoring_dataset, criterion, device, output_dir, epoch, total_epochs, accumulator, batch_size, num_workers, seed, log):
    """
    Compute LOW details for the scoring_dataset and accumulate values across epochs.
    Only writes final CSV at the last epoch.
    
    output_dir: folder where final CSV is saved
    epoch: integer epoch number (1-based)
    total_epochs: total number of epochs
    accumulator: dict to accumulate {image_id: {label, low_scores: [], sample_losses: [], loss_grad_norms: []}}
    Returns: updated accumulator
    """
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
                    "epoch": int(epoch),
                    "image_id": str(image_id),
                    "label": int(labels[batch_index].item()),
                    "sample_loss": float(loss_value.item()),
                    "loss_grad_norm": float(grad_value.item()),
                    "low_score": float(weight_value.item()),
                })

    # Update in-memory accumulator
    for r in rows:
        img = r['image_id']
        if img not in accumulator:
            accumulator[img] = {
                'label': r['label'],
                'low_scores': [],
                'sample_losses': [],
                'loss_grad_norms': []
            }
        accumulator[img]['low_scores'].append(r['low_score'])
        accumulator[img]['sample_losses'].append(r['sample_loss'])
        accumulator[img]['loss_grad_norms'].append(r['loss_grad_norm'])

    # Only write the final aggregated CSV after accumulation across all epochs
    if int(epoch) == int(total_epochs):
        os.makedirs(output_dir, exist_ok=True)
        agg_csv = os.path.join(output_dir, f'low_scores_aggregated_final_epoch_{epoch}.csv')
        with open(agg_csv, 'w', newline='') as csv_file:
            fieldnames = ['image_id', 'label', 'sum_low', 'count', 'avg_low', 'avg_sample_loss', 'avg_loss_grad_norm', 'last_low']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for img, vals in accumulator.items():
                lows = vals.get('low_scores', [])
                samples = vals.get('sample_losses', [])
                grads = vals.get('loss_grad_norms', [])
                count = len(lows)
                sum_low = float(sum(lows)) if count > 0 else 0.0
                avg_low = sum_low / count if count > 0 else 0.0
                avg_sample = float(sum(samples) / len(samples)) if len(samples) > 0 else 0.0
                avg_grad = float(sum(grads) / len(grads)) if len(grads) > 0 else 0.0
                last_low = float(lows[-1]) if count > 0 else 0.0
                writer.writerow({
                    'image_id': img,
                    'label': vals.get('label', 0),
                    'sum_low': round(sum_low, 6),
                    'count': count,
                    'avg_low': round(avg_low, 6),
                    'avg_sample_loss': round(avg_sample, 6),
                    'avg_loss_grad_norm': round(avg_grad, 6),
                    'last_low': round(last_low, 6),
                })
        log.info(f"Wrote final aggregated LOW CSV: {agg_csv}")
    else:
        log.info(f"Accumulated LOW scores for epoch {epoch}/{total_epochs}")
    
    return accumulator


def main():
    log = None
    try:
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
            log.info(f"Using device [{device}] to train model [{args.model_type}] on dataset [{args.dataset}] with [{'counterfactual augmentation' if args.use_counterfactuals or args.add_cf_batch else 'no counterfactuals'}] and [{'cross-validation' if args.cross_validation else 'no cross-validation'}].\n")

            # Create transforms (pass args so Mammo-CLIP branch uses upstream-style albumentations)
            train_transform, val_transform = create_transforms(
                args,
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
            
            # Prepare scoring dataset and accumulator for epoch-wise accumulation at seed level (before fold loop)
            scoring_dataset = None
            low_accumulator = {}  # In-memory accumulator for LOW scores
            if args.loss == 'low' and args.use_counterfactuals:
                # Create transforms for scoring
                _, val_transform = create_transforms(
                    augmentation_type=args.augmentation_type,
                    model_type=args.model_type
                )
                scoring_dataset = VinDrMammo_dataset(
                    split="train",
                    label="scoring",  # Special label to indicate this dataset is for scoring/accumulation only
                    cf_dir=args.cf_dir,
                    cv_fold=None,  # Use all samples regardless of fold
                    data_dir=args.data_dir,
                    metadata_path=args.metadata_path,
                    transform=val_transform,
                )
                log.info(f"LOW scoring dataset prepared at seed level with {len(scoring_dataset)} samples for accumulation across all folds.")
            
            if args.start_fold is not None:
                start_fold = args.start_fold
                if start_fold < 0 or start_fold >= folds:
                    raise ValueError(f"Invalid --start_fold value: {start_fold}. Must be between 0 and {folds-1}.")
            else:
                start_fold = 0
                
            for fold in range(start_fold, folds):
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
                #if device.type == 'cuda' and torch.cuda.device_count() > 1:
                    #model = nn.DataParallel(model)
                    #log.info(f"Enabled DataParallel across {torch.cuda.device_count()} CUDA devices")
                
                # Freeze image encoder blocks
                if args.model_type == "mammo-clip" or args.model_type == "fpn-mil":
                    if args.arch.lower().endswith("_lp"):
                        log.info("Linear probing selected via arch; image encoder remains frozen.")
                        unwrap_model(model).freeze_image_encoder()
                    else:
                        log.info("Finetuning selected via arch; image encoder isn't frozen.")
                else:
                    if args.freeze_layers > 0:
                        log.info(f"Freezing the first {args.freeze_layers} layers of the model as specified by --freeze_layers.")
                        unwrap_model(model).freeze_layers(args.freeze_layers)
                
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

                # Learning rate scheduler: Linear warmup + cosine annealing (upstream)
                # Determine warmup steps (expressed in optimizer steps / batches)
                total_steps = len(train_loader) * args.epochs
                we = float(getattr(args, 'warmup_epochs', 0.0))
                if we <= 0.0:
                    warmup_steps = 0
                elif we < 1.0:
                    # fractional fraction of total training steps (e.g. 0.1 means 10% of total steps)
                    warmup_steps = max(1, int(total_steps * we))
                else:
                    # convert nr of epochs to steps
                    warmup_steps = max(1, int(len(train_loader) * we))

                scheduler = LinearWarmupCosineAnnealingLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
                log.debug(f"LinearWarmupCosineAnnealingLR scheduler created. total_steps={total_steps}, warmup_steps={warmup_steps}")

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

                scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

                for epoch in range(start_epoch, args.epochs):
                    epoch_start_time = time.time()
                    log.info(f"\n{'='*15}Epoch {epoch+1}/{args.epochs}{'='*15}")

                    ###############################
                    # Removing unfreezing for now #   
                    ###############################   
                    # Gradual unfreezing for non-Mammo-CLIP and non-FPN-MIL models only.
                    #if args.model_type != "mammo-clip" and args.model_type != "fpn-mil" and (epoch == args.epochs // 4 or epoch == args.epochs // 2):  # Unfreeze after 25% and 50% of training
                        #unwrap_model(model).unfreeze_layers(epoch, args.epochs)

                    # Train
                    train_loss, train_acc, train_f1 = train_epoch(
                        model, train_loader, criterion, optimizer, device, scaler,
                        add_cf_batch=args.add_cf_batch,
                        cf_dir=args.cf_dir,
                        transform=train_transform,
                        use_mixup=args.use_mixup,
                        mixup_alpha=args.mixup_alpha,
                        scheduler=scheduler,
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
                            save_checkpoint(save_path, model, optimizer, epoch, args, val_loss=val_loss)
                            log.info(f"New best model saved. Val Loss: {val_loss:.4f}")
                        else:
                            patience_counter += 1
                            log.info(f"No improvement. Patience: {patience_counter}/{patience}")
                        if patience_counter >= patience:
                            log.info(f"Early stopping triggered after {epoch+1} epochs")
                            break
                    
                    epoch_end_time = time.time()
                    log.info(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")

                    # After epoch: compute LOW scores and accumulate (if configured)
                    if args.loss == 'low' and args.use_counterfactuals and scoring_dataset is not None:
                        low_accumulator = save_cf_low_scores(
                            model,
                            scoring_dataset,
                            criterion,
                            device,
                            seed_dir,
                            epoch+1,
                            args.epochs,
                            low_accumulator,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            seed=seed,
                            log=log,
                        )

                    if args.save_checkpoints > 0 and (epoch + 1) % args.save_checkpoints == 0 and epoch < args.epochs - 1:
                        save_path = os.path.join(fold_results_dir, f'checkpoint_epoch_{epoch+1}.pth')

                        save_checkpoint(save_path, model, optimizer, epoch, args)
                        log.info(f"Saved intermediate checkpoint at epoch {epoch+1} to: {save_path}")

                if args.no_validation:
                    # Save final model when no validation is used
                    save_path = os.path.join(fold_results_dir, 'final_model.pth')
                    save_checkpoint(save_path, model, optimizer, epoch, args)
                    log.info(f"Model trained for {args.epochs} epochs without validation and saved to: {save_path}")
                
                # Save training history
                with open(os.path.join(fold_results_dir, 'training_history.json'), 'w') as f:
                    json.dump(history, f, indent=2)

                # LOW scoring is done per-epoch during training when enabled.

                # Create training metrics plots
                if len(history['train_loss']) > 0:
                    plot_training_metrics(history, fold_results_dir, validation=(not args.no_validation), use_mixup=args.use_mixup)

                log.info(f"\n{'='*15}Training completed successfully!{'='*15}")
                if not args.no_validation:
                    log.info(f"Best validation loss: {best_val_loss:.4f}")
                else:
                    log.info("No validation was used; training-only history was saved.")
                log.info(f"Model and logs saved to: {fold_results_dir}")

    except Exception as exc:
        if log is not None:
            log.error(f"Training failed: {exc}")
        else:
            print(f"Training failed: {exc}", file=sys.stderr)
            print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), file=sys.stderr)
        raise




if __name__ == "__main__":
    main()