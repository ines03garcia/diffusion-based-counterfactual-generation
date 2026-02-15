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


def run_eval(run_path, args, device):

    if args.feature_extraction == 'online': 
        if 'efficientnetv2' in args.arch:
            args.model_base_name = 'efficientv2_s'
        elif 'efficientnet_b5_ns' in args.arch:
            args.model_base_name = 'efficientnetb5'
        else:
            args.model_base_name = args.arch
        
    args.n_class = 1 # Binary classification task
    args.resume= Path(args.resume)
    
    ############################ Data Setup ############################
    args.data_dir = Path(args.data_dir)
    
    args.df = pd.read_csv(args.csv_file)
    args.df = args.df.fillna(0)

    if args.eval_set == 'val': 
        dev_df = args.df[args.df['split'] == "training"].reset_index(drop=True)
        _, test_df = stratified_train_val_split(dev_df, 0.2, args = args)
    
    elif args.eval_set == 'test': # Use official test split
        print("Using test split")
        test_df = args.df[args.df['split'] == "test"].reset_index(drop=True)

    # Create DataLoader for MIL evaluation on test set
    test_loader = MIL_dataloader(test_df ,'test', args)
    print(f"Number of batches: {len(test_loader)}")
    print(f"Number of images in testing set: {len(test_loader)*args.batch_size}")

    # Build model
    model = build_model(args)
    model.is_training = False # Set model mode for evaluation
    
    model.to(device)
    #print_network(model)

    # Load best model checkpoint
    checkpoint = torch.load(os.path.join(run_path, 'best_model.pth'), map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # Get optimal threshold from checkpoint (use default 0.5 if not available)
    optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
    if 'optimal_threshold' in checkpoint:
        print(f"\nUsing optimal Youden threshold from checkpoint: {optimal_threshold:.4f}")
        if 'youden_j' in checkpoint:
            print(f"Youden's J statistic: {checkpoint['youden_j']:.4f}")
    else:
        print(f"\nOptimal threshold not found in checkpoint, using default: {optimal_threshold:.4f}")
    
    # Set the model to evaluation mode
    model.eval()

    test_targs, test_preds, test_probs, test_results = valid_fn(
        test_loader, model, criterion = torch.nn.BCEWithLogitsLoss(reduction='mean'), 
        args = args, device = device, split = 'test', threshold = optimal_threshold
    )
    
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
        
