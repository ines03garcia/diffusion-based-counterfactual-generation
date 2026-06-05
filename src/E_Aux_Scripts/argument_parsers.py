"""Argument parsers for classifier training and testing tasks."""

import argparse
import os

# Set paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
METADATA_ROOT = os.path.join(PROJECT_ROOT, "data/metadata")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data/images")


# ============================================================================
# TRAIN CLASSIFIER PARSER
# ============================================================================

def create_train_argparser():
    """Create argument parser for train_classifier.py"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['vindr'], default="vindr")
    parser.add_argument('--data_dir', type=str, default=os.path.join(IMAGES_ROOT, "VinDr-Mammo-Clip-CLAHE-512"))
    parser.add_argument('--metadata_path', type=str, default=os.path.join(METADATA_ROOT, "resized_df_512.json"))
    parser.add_argument('--training_category', type=str, choices=['all', 'healthy', 'anomalous', 'anomalous_with_findings'], default="all", help="Category of images to include in training")
    parser.add_argument('--cf_dir', type=str, default=os.path.join(IMAGES_ROOT, "repaint_results"))
    
    # Counterfactual augmentation mode (mutually exclusive)
    cf_aug_group = parser.add_mutually_exclusive_group()
    cf_aug_group.add_argument('--use_counterfactuals', action='store_true', default=False, help='Include counterfactual examples in the training dataset')
    cf_aug_group.add_argument('--add_cf_batch', action='store_true', default=False, help='Add corresponding counterfactuals to batches with their original images')
    parser.add_argument('--pair_loss_weight', type=float, default=0.0, help='Weight for counterfactual pair consistency loss when using --add_cf_batch')
    parser.add_argument('--pair_loss_type', type=str, choices=['kl', 'mse'], default='kl', help='Pair consistency loss type used in the joint objective')
    
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
    parser.add_argument('--single_seed', action='store_true', default=False, help='Train with a single seed (default mode)')
    parser.add_argument('--loss', type=str, choices=['bce', 'low'], default='bce', help="Loss function to use during training (default: Binary Cross-Entropy Loss with logits).")
    parser.add_argument("--save_checkpoints", type=int, default=0, help="Whether to save model checkpoints and how frequent (default: 0 to disable saving intermediate checkpoints).")

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
    mammo_clip_parser.add_argument("--data_frac", default=1.0, type=float, help="Fraction of data to be used for training")
    mammo_clip_parser.add_argument("--arch", default="breast_clip_det_b5_period_n_lp", type=str)

    mammo_clip_parser.add_argument("--swin_encoder", default="microsoft/swin-tiny-patch4-window7-224", type=str)
    mammo_clip_parser.add_argument("--pretrained_swin_encoder", default="y", type=str)
    mammo_clip_parser.add_argument("--swin_model_type", default="y", type=str)
    mammo_clip_parser.add_argument("--VER", default="084", type=str)
    mammo_clip_parser.add_argument("--alpha", default=10, type=float)
    mammo_clip_parser.add_argument("--sigma", default=15, type=float)
    mammo_clip_parser.add_argument("--p", default=1.0, type=float)
    mammo_clip_parser.add_argument("--mean", default=0.400409, type=float) # Calculated from the VinDrMammo-CLIP-CLAHE dataset after CLAHE
    mammo_clip_parser.add_argument("--std", default=0.259367, type=float)
    mammo_clip_parser.add_argument("--focal-alpha", default=0.6, type=float)
    mammo_clip_parser.add_argument("--focal-gamma", default=2.0, type=float)
    mammo_clip_parser.add_argument("--num-classes", default=1, type=int)
    mammo_clip_parser.add_argument("--num-workers", default=4, type=int)
    mammo_clip_parser.add_argument("--weight-decay", default=1e-4, type=float)
    mammo_clip_parser.add_argument("--warmup-epochs", default=1, type=float)
    mammo_clip_parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    mammo_clip_parser.add_argument("--device", default="cuda", type=str)
    mammo_clip_parser.add_argument("--apex", default="y", type=str)
    mammo_clip_parser.add_argument("--print-freq", default=5000, type=int)
    mammo_clip_parser.add_argument("--log-freq", default=1000, type=int)
    mammo_clip_parser.add_argument("--running-interactive", default='n', type=str)
    mammo_clip_parser.add_argument("--inference-mode", default='n', type=str)
    mammo_clip_parser.add_argument("--weighted-BCE", default='y', type=str)
    mammo_clip_parser.add_argument("--balanced-dataloader", default='n', type=str)

    fpn_mil_parser = subparsers.add_parser('fpn-mil')
    fpn_mil_parser.add_argument('--n_class', type=int, default=1, help='Number of classes for classification (default: 1 for binary classification)')
    fpn_mil_parser.add_argument('--batch_size', type=int, default=8)
    fpn_mil_parser.add_argument('--epochs', type=int, default=30)
    fpn_mil_parser.add_argument('--lr', type=float, default=5e-5)
    fpn_mil_parser.add_argument("--clip_chk_pt_path", default=os.path.join(MODELS_ROOT, "b2-model-best-epoch-10.tar"), type=str, help="Path to Mammo-CLIP chkpt")
    fpn_mil_parser.add_argument("--arch", default="upmc_vindr_breast_clip_det_b2_period_n_lp", type=str)

    # FPN-MIL: Patch extraction
    fpn_mil_parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    fpn_mil_parser.add_argument("--feature_extraction", default='online', type=str)
    fpn_mil_parser.add_argument("--feat_dim", default=352, type=int)
    fpn_mil_parser.add_argument('--patching', action='store_true', default=True, help='Whether to perform patching on full-resolution images. If false, it will consider previously extracted patches that were saved in a directory (default: True)')
    fpn_mil_parser.add_argument('--source_image', type=str, default='patches', choices=['patches', 'full_image'])
    fpn_mil_parser.add_argument('--patch_size', type=int, default=512)
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

    return parser


# ============================================================================
# TEST CLASSIFIER PARSER
# ============================================================================

def create_test_argparser():
    """Create argument parser for test_classifier.py"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["vindr", "inbreast"], default="vindr")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument(
        "--testing_category",
        type=str,
        choices=["all", "healthy", "anomalous", "anomalous_with_findings"],
        default="all",
        help="Category of images to include in testing (VinDr only)",
    )
    parser.add_argument("--cf_dir", type=str, default=os.path.join(IMAGES_ROOT, "repaint_results"))
    parser.add_argument(
        "--use_counterfactuals",
        action="store_true",
        default=False,
        help="Whether to include counterfactual examples during test loading (VinDr only)",
    )

    parser.add_argument("--cross-validation", action="store_true", default=False)
    parser.add_argument("--multiple_seeds", action="store_true", default=False, help="Evaluate all seed_i/final_model.pth checkpoints under --checkpoint_path")
    parser.add_argument("--aggregate_results", action="store_true", default=False, help="Aggregate metrics across folds after testing")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None, help="Path to existing test output folder to aggregate results (used with --multiple_seeds or --cross-validation)")
    parser.add_argument("--fixed_specificity", action="store_true", default=False, help="Evaluate the model at a fixed specificity operating point and skip the full metric suite")
    parser.add_argument("--fixed_specificity_value", type=float, default=0.80, help="Target specificity used when --fixed_specificity is enabled")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--loss', type=str, choices=['bce', 'low'], default='bce', help="Loss function to use during inference (default: Binary Cross-Entropy Loss with logits).")

    subparsers = parser.add_subparsers(dest="model_type", required=True)

    convnext_parser = subparsers.add_parser("convnext")
    convnext_parser.add_argument("--batch_size", type=int, default=16)

    vit_parser = subparsers.add_parser("vit")
    vit_parser.add_argument("--batch_size", type=int, default=16)

    mammo_clip_parser = subparsers.add_parser("mammo-clip")
    mammo_clip_parser.add_argument("--batch_size", type=int, default=8)
    mammo_clip_parser.add_argument(
        "--clip_chk_pt_path",
        default=os.path.join(MODELS_ROOT, "b5-model-best-epoch-7.tar"),
        type=str,
        help="Path to Mammo-CLIP checkpoint used to initialize encoder",
    )
    mammo_clip_parser.add_argument("--arch", default="upmc_vindr_breast_clip_det_b5_period_n_lp", type=str)

    fpn_mil_parser = subparsers.add_parser("fpn-mil")
    fpn_mil_parser.add_argument("--batch_size", type=int, default=8)
    fpn_mil_parser.add_argument("--n_class", type=int, default=1)
    fpn_mil_parser.add_argument("--clip_chk_pt_path", default=os.path.join(MODELS_ROOT, "b2-model-best-epoch-10.tar"), type=str)
    fpn_mil_parser.add_argument("--arch", default="upmc_vindr_breast_clip_det_b2_period_n_lp", type=str)
    fpn_mil_parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    fpn_mil_parser.add_argument("--feature_extraction", default="online", type=str)
    fpn_mil_parser.add_argument("--feat_dim", default=352, type=int)
    fpn_mil_parser.add_argument("--patching", action="store_true", default=True)
    fpn_mil_parser.add_argument("--source_image", type=str, default="patches", choices=["patches", "full_image"])
    fpn_mil_parser.add_argument("--patch_size", type=int, default=512)
    fpn_mil_parser.add_argument("--overlap", type=float, nargs="*", default=[0.0])
    fpn_mil_parser.add_argument("--mil_type", default="pyramidal_mil", choices=[None, "instance", "embedding", "pyramidal_mil"], type=str)
    fpn_mil_parser.add_argument("--pooling_type", default="gated-attention", choices=["max", "mean", "attention", "gated-attention", "pma"], type=str)
    fpn_mil_parser.add_argument("--type_mil_encoder", default="mlp", choices=["mlp", "sab", "isab"], type=str)
    fpn_mil_parser.add_argument("--fcl_attention_dim", type=int, default=128)
    fpn_mil_parser.add_argument("--map_prob_func", type=str, default="softmax", choices=["softmax", "sparsemax", "entmax", "alpha_entmax"])
    fpn_mil_parser.add_argument("--fcl_encoder_dim", type=int, default=256)
    fpn_mil_parser.add_argument("--sab_num_heads", type=int, default=4)
    fpn_mil_parser.add_argument("--isab_num_heads", type=int, default=4)
    fpn_mil_parser.add_argument("--pma_num_heads", type=int, default=1)
    fpn_mil_parser.add_argument("--num_encoder_blocks", type=int, default=2)
    fpn_mil_parser.add_argument("--trans_num_inds", type=int, default=20)
    fpn_mil_parser.add_argument("--trans_layer_norm", type=bool, default=False)
    fpn_mil_parser.add_argument("--multi_scale_model", type=str, choices=["fpn", "backbone_pyramid", "msp"], default="fpn")
    fpn_mil_parser.add_argument("--scales", type=int, nargs="*", default=(16, 32, 128))
    fpn_mil_parser.add_argument("--fpn_dim", type=int, default=256)
    fpn_mil_parser.add_argument("--upsample_method", type=str, choices=["bilinear", "nearest"], default="nearest")
    fpn_mil_parser.add_argument("--norm_fpn", type=bool, default=False)
    fpn_mil_parser.add_argument("--deep_supervision", action="store_true", default=True)
    fpn_mil_parser.add_argument("--type_scale_aggregator", type=str, choices=["concatenation", "max_p", "mean_p", "attention", "gated-attention"], default="gated-attention")
    fpn_mil_parser.add_argument("--drop_classhead", type=float, default=0.0)
    fpn_mil_parser.add_argument("--drop_attention_pool", type=float, default=0.0)
    fpn_mil_parser.add_argument("--drop_mha", type=float, default=0.0)
    fpn_mil_parser.add_argument("--fcl_dropout", type=float, default=0.0)
    fpn_mil_parser.add_argument("--lamda", type=float, default=0.0)
    fpn_mil_parser.add_argument("--nested_model", action="store_true", default=False)

    return parser


# ============================================================================
# TEST CLASSIFIER SIGNIFICANCE PARSER
# ============================================================================

DEFAULT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "roc_auc",
    "log_loss",
]


def create_significance_argparser():
    """Create argument parser for test_classifier_significance.py"""
    parser = argparse.ArgumentParser(description="Compare two classifier result directories with ASO.")
    parser.add_argument("--baseline_aug", required=True, help="Directory or test_metrics.json file for baseline augmentation runs.")
    parser.add_argument("--cf_aug", required=True, help="Directory or test_metrics.json file for counterfactual augmentation runs.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        type=str,
        default=DEFAULT_METRICS,
        help="Metrics to compare from test_metrics.json. Defaults to all standard classification metrics.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed passed to ASO.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional JSON file to save the comparison result.")
    return parser
