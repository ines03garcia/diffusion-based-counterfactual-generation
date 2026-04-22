import argparse
import json
import math
import os
import statistics
import sys
import torch
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
METADATA_ROOT = os.path.join(PROJECT_ROOT, "data/metadata")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data/images")

from src.E_Aux_Scripts import logger
from src.E_Aux_Scripts.utils import set_seed, seed_worker
from src.E_Aux_Scripts.classifier_helpers import (
	build_model,
	create_transforms,
    get_test_dataset,
    load_checkpoint_weights,
    run_inference,
    compute_metrics,
    save_confusion_matrix,
    save_roc_curve
)


def main():
	parser = create_argparser()
	args = parser.parse_args()

	# Validate checkpoint arguments
	if args.multiple_seeds:
		if args.cross_validation:
			raise ValueError("--multiple_seeds cannot be used with --cross-validation.")
		if args.checkpoint_dir is not None:
			raise ValueError("--checkpoint_dir cannot be used with --multiple_seeds. Use --checkpoint_path pointing to the parent directory.")
		if args.checkpoint_path is None:
			raise ValueError("--checkpoint_path is required with --multiple_seeds.")
	else:
		if args.cross_validation:
			if args.checkpoint_path is not None:
				raise ValueError("--checkpoint_path cannot be used with --cross-validation. Use --checkpoint_dir instead.")
			if args.checkpoint_dir is None:
				raise ValueError("--checkpoint_dir is required when using --cross-validation.")
		else:
			if args.checkpoint_path is None and args.checkpoint_dir is None:
				raise ValueError("Either --checkpoint_path or --checkpoint_dir must be provided.")

	if args.data_dir is None:
		if args.dataset == "vindr":
			args.data_dir = os.path.join(IMAGES_ROOT, "VinDr-Mammo-Clip-CLAHE-512")
		else:
			args.data_dir = os.path.join(IMAGES_ROOT, "Inbreast_png")

	if args.metadata_path is None:
		if args.dataset == "vindr":
			args.metadata_path = os.path.join(METADATA_ROOT, "resized_df_512.json")
		else:
			args.metadata_path = os.path.join(METADATA_ROOT, "inbreast_test_metadata.csv")

	set_seed(args.seed)

	setup = f"{args.dataset}_{'with_cf' if args.use_counterfactuals else 'no_cf'}"
	log = logger.Logger(experiment_type="Classifiers", sub_experiment_type="test", model_type=args.model_type, setup=setup)
	log.configure_root_logger()
	log.info(f"Logs will be saved to: {log.output_dir}")

	# Save testing args.
	args_output_path = os.path.join(log.output_dir, "test_args.json")
	with open(args_output_path, "w") as f:
		json.dump(vars(args), f, indent=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	log.info(f"Using device [{device}] for testing model [{args.model_type}] on dataset [{args.dataset}]")

	_, test_transform = create_transforms(
		augmentation_type="none",
		model_type=args.model_type,
	)

	test_dataset = get_test_dataset(args, test_transform)
	class_dist = test_dataset.get_class_distribution()
	log.info(f"Loaded {len(test_dataset)} test samples with class distribution {class_dist}")

	test_loader = DataLoader(
		test_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		worker_init_fn=lambda x: seed_worker(args, x),
		generator=torch.Generator().manual_seed(args.seed),
	)

	model = build_model(args, device, experiment="test")

	seed_checkpoints = []
	if args.multiple_seeds:
		seed_root = args.checkpoint_path
		if not os.path.isabs(seed_root):
			# Prefer paths relative to the project root (e.g. data/logs/...),
			# then fall back to models/ for backward compatibility.
			project_relative = os.path.join(PROJECT_ROOT, seed_root)
			models_relative = os.path.join(MODELS_ROOT, seed_root)
			if os.path.isdir(project_relative):
				seed_root = project_relative
			else:
				seed_root = models_relative
		if not os.path.isdir(seed_root):
			raise ValueError("With --multiple_seeds, --checkpoint_path must be a directory containing seed_i folders.")

		seed_dirs = sorted(
			[
				d for d in os.listdir(seed_root)
				if d.startswith("seed_") and os.path.isdir(os.path.join(seed_root, d))
			]
		)
		for seed_dir in seed_dirs:
			seed_ckpt = os.path.join(seed_root, seed_dir, "final_model.pth")
			if os.path.exists(seed_ckpt):
				seed_checkpoints.append((seed_dir, seed_ckpt))
			else:
				log.warning(f"Skipping {seed_dir}: missing checkpoint {seed_ckpt}")

		if not seed_checkpoints:
			raise FileNotFoundError(f"No seed_i/final_model.pth checkpoints found under {seed_root}")

		num_runs = len(seed_checkpoints)
	elif args.cross_validation:
		num_runs = 4
	else:
		num_runs = 1

	fold_metrics_list = []

	for run_idx in range(num_runs):
		# Resolve checkpoint path
		seed_name = None
		if args.multiple_seeds:
			seed_name, checkpoint_path = seed_checkpoints[run_idx]
			log.info(f"Loading checkpoint from seed directory: {seed_name}")
		elif args.checkpoint_path is not None:
			checkpoint_path = args.checkpoint_path
			if not os.path.isabs(checkpoint_path):
				checkpoint_path = os.path.join(MODELS_ROOT, checkpoint_path)
		else:
			# checkpoint_dir is provided
			checkpoint_dir = args.checkpoint_dir
			if not os.path.isabs(checkpoint_dir):
				checkpoint_dir = os.path.join(MODELS_ROOT, checkpoint_dir)
			
			if num_runs > 1:
				checkpoint_dir = os.path.join(checkpoint_dir, f"fold_{run_idx}")
				log.info(f"Loading checkpoint from directory: {checkpoint_dir}")
			
			checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
		
		load_checkpoint_weights(model, checkpoint_path, device)
		if args.multiple_seeds:
			log.info(f"Starting testing for seed {seed_name} on the full test set...")
		elif args.cross_validation:
			log.info(f"Starting testing for fold {run_idx}...")
		else:
			log.info("Starting testing on the full test set...")
		probs, preds, targets, image_ids = run_inference(model, test_loader, device)
		metrics, cm, fpr, tpr, roc_auc = compute_metrics(targets, preds, probs)

		metrics["dataset"] = args.dataset
		metrics["model_type"] = args.model_type
		metrics["checkpoint_path"] = checkpoint_path

		if args.multiple_seeds:
			metrics["seed"] = seed_name
			log_dir = os.path.join(log.output_dir, seed_name)
			os.makedirs(log_dir, exist_ok=True)
		elif num_runs > 1:
			metrics["fold"] = run_idx
			log_dir = os.path.join(log.output_dir, f"fold_{run_idx}")
			os.makedirs(log_dir, exist_ok=True)
			fold_metrics_list.append(metrics)
		else:
			log_dir = log.output_dir

		cm_path = os.path.join(log_dir, "confusion_matrix.png")
		roc_path = os.path.join(log_dir, "roc_curve.png")
		save_confusion_matrix(cm, cm_path)
		save_roc_curve(fpr, tpr, roc_auc, roc_path)

		metrics_output_path = os.path.join(log_dir, "test_metrics.json")
		with open(metrics_output_path, "w") as f:
			json.dump(metrics, f, indent=2)

		preds_output_path = os.path.join(log_dir, "predictions.json")
		predictions_payload = [
			{
				"image_id": image_id,
				"target": int(target),
				"pred": int(pred),
				"prob": float(prob),
			}
			for image_id, target, pred, prob in zip(image_ids, targets, preds, probs)
		]
		with open(preds_output_path, "w") as f:
			json.dump(predictions_payload, f, indent=2)

		log.info("Testing completed successfully")
		log.info(f"Metrics: {json.dumps(metrics, indent=2)}")
		log.info(f"Saved args to: {args_output_path}")
		log.info(f"Saved metrics to: {metrics_output_path}")
		log.info(f"Saved confusion matrix to: {cm_path}")
		log.info(f"Saved ROC curve to: {roc_path}")
		log.info(f"Saved predictions to: {preds_output_path}")

	# Aggregate results across folds if requested
	if args.cross_validation and args.aggregate_results and fold_metrics_list:
		log.info("Aggregating results across folds...")
		aggregate_dir = os.path.join(log.output_dir, "aggregated")
		os.makedirs(aggregate_dir, exist_ok=True)

		percentage_metrics = {
			"accuracy",
			"balanced_accuracy",
			"precision",
			"recall",
			"f1_score",
			"specificity",
			"roc_auc",
		}
		metric_baselines = {
			"accuracy": 50.0,
			"balanced_accuracy": 50.0,
			"precision": 50.0,
			"recall": 50.0,
			"f1_score": 50.0,
			"specificity": 50.0,
			"roc_auc": 50.0,
			"log_loss": math.log(2.0),
		}
		
		# Collect all metrics keys (excluding fold, dataset, model_type, checkpoint_path)
		metric_keys = set()
		for fold_metrics in fold_metrics_list:
			for key in fold_metrics.keys():
				if key not in ["fold", "dataset", "model_type", "checkpoint_path", "num_samples"]:
					if isinstance(fold_metrics[key], (int, float)):
						metric_keys.add(key)
		
		# Calculate means and stds
		aggregated_metrics = {
			"dataset": args.dataset,
			"model_type": args.model_type,
			"num_folds": num_runs,
		}
		
		for metric_key in sorted(metric_keys):
			values = [fold_metrics[metric_key] for fold_metrics in fold_metrics_list if metric_key in fold_metrics]
			if values:
				mean_value = sum(values) / len(values)
				std_value = statistics.stdev(values) if len(values) > 1 else 0.0
				if metric_key in percentage_metrics:
					aggregated_metrics[metric_key] = f"{mean_value:.1f} +- {std_value:.2f}"
				elif metric_key == "log_loss":
					aggregated_metrics[metric_key] = f"{mean_value:.3f} +- {std_value:.3f}"
				else:
					aggregated_metrics[metric_key] = f"{mean_value:.3f} +- {std_value:.3f}"
		
		# Save aggregated metrics
		agg_metrics_path = os.path.join(aggregate_dir, "test_metrics.json")
		with open(agg_metrics_path, "w") as f:
			json.dump(aggregated_metrics, f, indent=2)
		
		log.info(f"Aggregated metrics saved to: {agg_metrics_path}")
		log.info(f"Aggregated metrics: {json.dumps(aggregated_metrics, indent=2)}")


def create_argparser():
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
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--pretrained", action="store_true", default=True)
	parser.add_argument("--seed", type=int, default=0)

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
	fpn_mil_parser.add_argument("--img-size", default=[512, 512], type=int, nargs="*")
	fpn_mil_parser.add_argument("--feature_extraction", default="online", type=str)
	fpn_mil_parser.add_argument("--feat_dim", default=352, type=int)
	fpn_mil_parser.add_argument("--patching", action="store_true", default=True)
	fpn_mil_parser.add_argument("--source_image", type=str, default="patches", choices=["patches", "full_image"])
	fpn_mil_parser.add_argument("--patch_size", type=int, default=128)
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


if __name__ == "__main__":
	main()
