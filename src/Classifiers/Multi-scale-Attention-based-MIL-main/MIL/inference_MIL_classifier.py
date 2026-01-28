import numpy as np
import pandas as pd
from pathlib import Path
import os 

import torch
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score, 
                             precision_score, recall_score, confusion_matrix)

from Datasets.dataset_utils import MIL_dataloader
from MIL import build_model 
from MIL.MIL_experiment import valid_fn
from utils.generic_utils import seed_all, print_network
from utils.plot_utils import plot_confusion_matrix, ROC_curves
from utils.data_split_utils import stratified_train_val_split


def setup_model_args(args):
    """Setup model-specific arguments based on architecture."""
    if args.feature_extraction == 'online': 
        if 'efficientnetv2' in args.arch:
            args.model_base_name = 'efficientv2_s'
        elif 'efficientnet_b5_ns' in args.arch:
            args.model_base_name = 'efficientnetb5'
        else:
            args.model_base_name = args.arch
    
    args.n_class = 1  # Binary classification task
    return args


def load_and_prepare_data(args):
    """Load and prepare test data based on evaluation set."""
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    
    print(f"df shape: {args.df.shape}")
    print(args.df.columns)

    if args.eval_set == 'val': 
        dev_df = args.df[args.df['split'] == "training"].reset_index(drop=True)
        _, test_df = stratified_train_val_split(dev_df, 0.2, args=args)
    elif args.eval_set == 'test':
        test_df = args.df[args.df['split'] == "test"].reset_index(drop=True)
    
    return MIL_dataloader(test_df, 'test', args)


def load_checkpoint_and_inspect(model, run_path, device):
    """Load model checkpoint and inspect weights."""
    checkpoint_path = os.path.join(run_path, 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    
    if missing_keys:
        print(f"WARNING: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"WARNING: Unexpected keys in checkpoint: {unexpected_keys}")
    
    print(f"\n=== Model Weight Debug ===")
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    # Check main classifier
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'head_classifier'):
        classifier_weight = model.classifier.head_classifier.weight.data
        classifier_bias = model.classifier.head_classifier.bias.data if model.classifier.head_classifier.bias is not None else None
        print(f"Classifier weight shape: {classifier_weight.shape}, mean: {classifier_weight.mean():.4f}, std: {classifier_weight.std():.4f}")
        if classifier_bias is not None:
            print(f"Classifier bias: {classifier_bias.item():.4f}")
    
    # Check side classifiers (deep supervision)
    if hasattr(model, 'side_classifiers'):
        for key in model.side_classifiers.keys():
            if hasattr(model.side_classifiers[key], 'head_classifier'):
                bias = model.side_classifiers[key].head_classifier.bias
                if bias is not None:
                    print(f"{key} bias: {bias.data.item():.4f}")
    
    return model


def print_scale_metrics(test_results, args):
    """Print per-scale and aggregated metrics."""
    print(f"\nTest Loss: {test_results['loss']:.4f}")
    
    for s in args.scales:
        print(f"Scale: {s} --> Test F1-Score: {test_results[s]['f1']:.4f} | "
              f"Test Bacc: {test_results[s]['bacc']:.4f} | "
              f"Test ROC-AUC: {test_results[s]['auc_roc']:.4f}")
    
    print(f"Aggregated Results --> Test F1-Score: {test_results['aggregated']['f1']:.4f} | "
          f"Test Bacc: {test_results['aggregated']['bacc']:.4f} | "
          f"Test ROC-AUC: {test_results['aggregated']['auc_roc']:.4f}")


def analyze_thresholds(test_targs, test_probs, test_preds):
    """Perform threshold analysis to find optimal operating point."""
    cm = confusion_matrix(test_targs, test_preds)
    
    print(f"\nConfusion Matrix (Aggregated) with threshold=0.5:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"\nPredictions: {test_preds.sum()}/{len(test_preds)} positive ({100*test_preds.sum()/len(test_preds):.1f}%)")
    print(f"Ground Truth: {test_targs.sum()}/{len(test_targs)} positive ({100*test_targs.sum()/len(test_targs):.1f}%)")
    print(f"Probability range: [{test_probs.min():.3f}, {test_probs.max():.3f}], mean: {test_probs.mean():.3f}")
    
    print(f"\n=== Threshold Analysis ===")
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        preds_at_threshold = (test_probs >= threshold).astype(int)
        f1 = f1_score(test_targs, preds_at_threshold)
        bacc = balanced_accuracy_score(test_targs, preds_at_threshold)
        num_pos = preds_at_threshold.sum()
        
        print(f"Threshold={threshold:.2f}: F1={f1:.4f}, Bacc={bacc:.4f}, "
              f"Pos predictions={num_pos}/{len(preds_at_threshold)} ({100*num_pos/len(preds_at_threshold):.1f}%)")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} with F1={best_f1:.4f}")
    return best_threshold


def compute_and_print_metrics(test_targs, test_probs, optimal_threshold, auc):
    """Compute and print all evaluation metrics with optimal threshold."""
    preds_optimal = (test_probs >= optimal_threshold).astype(int)
    
    # Calculate all metrics
    accuracy = accuracy_score(test_targs, preds_optimal)
    precision = precision_score(test_targs, preds_optimal, zero_division=0)
    recall = recall_score(test_targs, preds_optimal, zero_division=0)
    f1_optimal = f1_score(test_targs, preds_optimal, zero_division=0)
    bacc_optimal = balanced_accuracy_score(test_targs, preds_optimal)
    
    # Calculate specificity
    cm_optimal = confusion_matrix(test_targs, preds_optimal)
    tn, fp, fn, tp = cm_optimal.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f"\n=== Results with Optimal Threshold ({optimal_threshold}) ===")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1_optimal:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Bacc:        {bacc_optimal:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")


def run_eval(run_path, args, device):
    """Run evaluation on test set for a single model checkpoint."""
    # Setup
    args = setup_model_args(args)
    args.resume = Path(args.resume)

    # Load data and create dataloader
    test_loader = load_and_prepare_data(args)

    # Build and load model
    model = build_model(args)
    model.is_training = False
    model.to(device)
    print_network(model)

    # Load checkpoint and inspect weights
    model = load_checkpoint_and_inspect(model, run_path, device)
    model.eval()

    # Run inference
    test_targs, test_preds, test_probs, test_results = valid_fn(
        test_loader, model, 
        criterion=torch.nn.BCEWithLogitsLoss(reduction='mean'), 
        args=args, device=device, split='test'
    )
    
    # Print results
    print_scale_metrics(test_results, args)
    
    # Threshold analysis and final metrics
    if 'cf_matrix' in test_results['aggregated'] and test_results['aggregated']['cf_matrix'] is not None:
        best_threshold = analyze_thresholds(test_targs, test_probs, test_preds)
        compute_and_print_metrics(test_targs, test_probs, best_threshold, 
                                  test_results['aggregated']['auc_roc'])
    
    # Prepare results dataframe
    final_results_data = {}
    for s in args.scales:
        final_results_data[f'{args.eval_set}_bacc_{s}'] = test_results[s]['bacc']
        final_results_data[f'{args.eval_set}_f1_{s}'] = test_results[s]['f1']
        final_results_data[f'{args.eval_set}_auc_roc_{s}'] = test_results[s]['auc_roc']
    
    final_results_data[f'{args.eval_set}_bacc_aggregated'] = test_results['aggregated']['bacc']
    final_results_data[f'{args.eval_set}_f1_aggregated'] = test_results['aggregated']['f1']
    final_results_data[f'{args.eval_set}_auc_roc_aggregated'] = test_results['aggregated']['auc_roc']
    
    return pd.DataFrame(final_results_data, index=[0])


def Eval(args, device):

    all_results = []  # Store results from all runs

    for run_idx in range(args.n_runs):
        seed_all(args.seed)
        
        print(f'\nRunning eval for model run nº{run_idx + args.start_run}....')
        
        run_path = os.path.join(args.resume, f'run_{args.start_run + run_idx}')
        
        # Run the evaluation and get results as DataFrame
        run_results_df = run_eval(run_path, args, device) 
        
        # Add column to track the run
        run_results_df["runs"] = args.start_run + run_idx
        
        all_results.append(run_results_df)
    
    if args.n_runs > 1: 

        # Combine all runs into a single DataFrame
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Calculate mean and std for specific columns
        mean_std = combined_df.drop('runs', axis=1).agg(['mean', 'std']).reset_index(drop=True)
        mean_std['runs'] = ['mean', 'std']

        # Append mean and std to the original DataFrame
        combined_df = pd.concat([combined_df, mean_std]).reset_index(drop=True)

        print(combined_df)
    
        output_path = os.path.join(args.resume, f'{args.dataset}_eval_summary.csv')
        combined_df.to_csv(output_path, index=False)
        
