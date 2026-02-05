import numpy as np
import pandas as pd
from pathlib import Path
import os 
import json

import torch

from Datasets.dataset_utils import MIL_dataloader
from MIL import build_model 
from MIL.MIL_experiment import valid_fn
from utils.generic_utils import seed_all, print_network
from utils.plot_utils import plot_confusion_matrix, ROC_curves
from utils.data_split_utils import stratified_train_val_split
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score, 
                             precision_score, recall_score, confusion_matrix)

def compute_and_print_metrics(test_targs, test_probs, threshold, auc):
    """Compute and print all evaluation metrics with given threshold."""
    preds_optimal = (test_probs >= threshold).astype(int)
    
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
    
    print(f"\n=== Results with Threshold ({threshold}) ===")
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
    
    # Return metrics as dictionary
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_optimal,
        'specificity': specificity,
        'balanced_accuracy': bacc_optimal,
        'roc_auc': auc,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    return metrics

def run_eval(run_path, args, device):

    if args.feature_extraction == 'online': 
        if 'efficientnetv2' in args.arch:
            args.model_base_name = 'efficientv2_s'
        elif 'efficientnet_b5_ns' in args.arch:
            args.model_base_name = 'efficientnetb5'
        else:
            args.model_base_name = args.arch
        
    args.n_class = 1 # Binary classification task

    # Define class labels 
    if args.label.lower() == 'mass':
        class0 = 'not_mass'
        class1 = 'mass'
    elif args.label.lower() == 'suspicious_calcification':
        class0 = 'not_calcification'
        class1 = 'calcification'
    elif args.label.lower() == 'anomaly':
        class0 = 'healthy'
        class1 = 'anomalous'

    label_dict = {class0: 0, class1: 1}

    args.resume= Path(args.resume)
    
    ############################ Data Setup ############################
    args.data_dir = Path(args.data_dir)
    
    args.df = pd.read_csv(args.csv_file)
    args.df = args.df.fillna(0)
    
    print(f"df shape: {args.df.shape}")
    print(args.df.columns)

    if args.eval_set == 'val': 
        dev_df = args.df[args.df['split'] == "training"].reset_index(drop=True)
        _, test_df = stratified_train_val_split(dev_df, 0.2, args = args)
    
    elif args.eval_set == 'test': # Use official test split
        test_df = args.df[args.df['split'] == "test"].reset_index(drop=True)

    # Create DataLoader for MIL evaluation on test set
    test_loader = MIL_dataloader(test_df ,'test', args)

    # Build model
    model = build_model(args)
    model.is_training = False # Set model mode for evaluation
    
    model.to(device)
    print_network(model)

    # Load best model checkpoint
    checkpoint = torch.load(os.path.join(run_path, 'best_model.pth'), map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # Set the model to evaluation mode
    model.eval()

    test_targs, test_preds, test_probs, test_results = valid_fn(
        test_loader, model, criterion = torch.nn.BCEWithLogitsLoss(reduction='mean'), args = args, device = device, split = 'test'
    )
    
    # Print overall test loss
    print(f"\nTest Loss: {test_results['loss']:.4f}")     

    # Print metrics per scale
    for s in args.scales:
        print(f"Scale: {s} --> Test F1-Score: {test_results[s]['f1']:.4f} | Test Bacc: {test_results[s]['bacc']:.4f} | Test ROC-AUC: {test_results[s]['auc_roc']:.4f}")            

    # Print aggregated metrics across scales
    print(f"Aggregated Results --> Test F1-Score: {test_results['aggregated']['f1']:.4f} | Test Bacc: {test_results['aggregated']['bacc']:.4f} | Test ROC-AUC: {test_results['aggregated']['auc_roc']:.4f}")
        
    metrics = None
    if 'cf_matrix' in test_results['aggregated'] and test_results['aggregated']['cf_matrix'] is not None:
        metrics = compute_and_print_metrics(test_targs, test_probs, 0.5, 
                                           test_results['aggregated']['auc_roc'])
    
    if metrics is not None:
        # Round float metrics to 4 decimal places
        rounded_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                rounded_metrics[key] = round(value, 4)
            else:
                rounded_metrics[key] = value
        
        # Save metrics as JSON
        metrics_path = os.path.join(args.output_dir, f'{args.eval_set}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(rounded_metrics, f, indent=4)
        print(f"\nSaved {args.eval_set} metrics to: {metrics_path}")
    
    final_results_data = {}
    
    # Append metrics for all scales
    for s in args.scales:
        final_results_data[f'{args.eval_set}_bacc_{s}'] = test_results[s]['bacc']
        final_results_data[f'{args.eval_set}_f1_{s}'] = test_results[s]['f1']
        final_results_data[f'{args.eval_set}_auc_roc_{s}'] = test_results[s]['auc_roc']
        
    # Append metrics for aggregated results
    final_results_data[f'{args.eval_set}_bacc_aggregated'] = test_results['aggregated']['bacc']
    final_results_data[f'{args.eval_set}_f1_aggregated'] = test_results['aggregated']['f1']
    final_results_data[f'{args.eval_set}_auc_roc_aggregated'] = test_results['aggregated']['auc_roc']
        
    # Create the final DataFrame
    df_final_results = pd.DataFrame(final_results_data, index=[0])

    return df_final_results


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
    
        output_path = os.path.join(args.output_path, f'{args.dataset}_eval_summary.csv')
        combined_df.to_csv(output_path, index=False)
        
