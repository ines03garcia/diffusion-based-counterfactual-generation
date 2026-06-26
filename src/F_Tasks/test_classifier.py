import json
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
	compute_fixed_specificity_metrics,
    save_confusion_matrix,
    save_roc_curve
)
from src.E_Aux_Scripts.argument_parsers import create_test_argparser


def fixed_specificity_suffix(fixed_specificity, fixed_specificity_value):
	if not fixed_specificity:
		return ""
	return f"_fixed_specificity_{fixed_specificity_value:.2f}".replace(".", "p")


def aggregate_existing_outputs(output_path, multiple_seeds, cross_validation, project_root, fixed_specificity=False, fixed_specificity_value=0.80):
	"""Aggregate existing test outputs under `output_path` (seed_/fold_ subfolders).

	Returns the path to the aggregated output file written.
	"""
	if not (multiple_seeds or cross_validation):
		raise ValueError("--output_path requires --multiple_seeds or --cross-validation to aggregate results")

	if not os.path.isabs(output_path):
		output_path = os.path.join(project_root, output_path)

	if not os.path.isdir(output_path):
		raise ValueError(f"Provided --output_path does not exist or is not a directory: {output_path}")

	# Discover child result folders
	child_prefix = "seed_" if multiple_seeds else "fold_"
	child_dirs = sorted([
		d for d in os.listdir(output_path)
		if d.startswith(child_prefix) and os.path.isdir(os.path.join(output_path, d))
	])
	if not child_dirs:
		raise FileNotFoundError(f"No '{child_prefix}*' subdirectories found under {output_path}")

	metrics_list = []
	for d in child_dirs:
		metrics_file = os.path.join(output_path, d, "test_metrics.json")
		if not os.path.exists(metrics_file):
			print(f"Skipping {d}: missing {metrics_file}")
			continue
		with open(metrics_file, 'r') as fh:
			metrics_list.append(json.load(fh))

	if not metrics_list:
		raise FileNotFoundError(f"No test_metrics.json files found under {output_path}")

	# Aggregate numeric metrics (mean ± std)
	agg_metrics = {}
	for metric_name in metrics_list[0].keys():
		if metric_name in ["dataset", "model_type", "checkpoint_path"]:
			continue
		metric_values = [m.get(metric_name) for m in metrics_list if metric_name in m]
		if all(isinstance(v, (int, float)) for v in metric_values):
			if metric_name == "num_samples":
				agg_metrics[metric_name] = list(set(metric_values))
				continue
			agg_metrics[metric_name] = {
				"mean": statistics.mean(metric_values),
				"std": round(statistics.stdev(metric_values), 4) if len(metric_values) > 1 else 0.0,
			}
		else:
			# skip non-numeric metrics
			continue

	agg_output_path = os.path.join(
		output_path,
		f"aggregated_seed_metrics{fixed_specificity_suffix(fixed_specificity, fixed_specificity_value)}.json",
	)
	with open(agg_output_path, 'w') as fh:
		json.dump(agg_metrics, fh, indent=2)
	print(f"Aggregated metrics saved to: {agg_output_path}")
	return agg_output_path


def main():
	parser = create_test_argparser()
	args = parser.parse_args()

	# If test output folder is provided, aggregate results and exit
	if args.output_path is not None:
		aggregate_existing_outputs(
			args.output_path,
			args.multiple_seeds,
			args.cross_validation,
			PROJECT_ROOT,
			fixed_specificity=args.fixed_specificity,
			fixed_specificity_value=args.fixed_specificity_value,
		)
		return

	if args.fixed_specificity and not (0.0 < args.fixed_specificity_value <= 1.0):
		raise ValueError("--fixed_specificity_value must be in the interval (0, 1].")

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
			args.data_dir = os.path.join(IMAGES_ROOT, "datasets/Inbreast_png")

	if args.metadata_path is None:
		if args.dataset == "vindr":
			args.metadata_path = os.path.join(METADATA_ROOT, "resized_df_512.json")
		else:
			args.metadata_path = os.path.join(METADATA_ROOT, "inbreast_test_metadata.csv")

	set_seed(args.seed)

	if args.multiple_seeds:
		config = "/multiple_seeds"
	elif args.cross_validation:
		config = "/cross_validation"
	else:
		config = "/"

	cp_norm = os.path.normpath(args.checkpoint_path)
	cp_parts = cp_norm.split(os.path.sep)

	if 'cf_aug' in cp_parts:
		aug_tag = '/cf_aug'
	elif 'baseline_aug' in cp_parts:
		aug_tag = '/baseline_aug'
	else:
		aug_tag = '/unknown_aug'

	setup = f"{config}{aug_tag}"
	log = logger.Logger(experiment_type="Classifiers", sub_experiment_type="test", label=args.label, dataset=args.dataset, model_type=args.model_type, setup=setup)
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

	metrics_list = []

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
		probs, preds, targets, image_ids = run_inference(model, test_loader, device, threshold=args.threshold)
		if args.fixed_specificity:
			metrics, cm, preds = compute_fixed_specificity_metrics(targets, probs, args.fixed_specificity_value)
			fpr = None
			tpr = None
			roc_auc = float("nan")
			log.info(
				f"Using fixed specificity target {args.fixed_specificity_value:.2f} with decision threshold {metrics['decision_threshold']:.6f}"
			)
		else:
			metrics, cm, fpr, tpr, roc_auc = compute_metrics(targets, preds, probs)

		metrics["dataset"] = args.dataset
		metrics["model_type"] = args.model_type
		metrics["checkpoint_path"] = checkpoint_path

		if args.multiple_seeds:
			metrics["seed"] = seed_name
			log_dir = os.path.join(log.output_dir, seed_name)
			os.makedirs(log_dir, exist_ok=True)
			metrics_list.append(metrics)
		elif num_runs > 1:
			metrics["fold"] = run_idx
			log_dir = os.path.join(log.output_dir, f"fold_{run_idx}")
			os.makedirs(log_dir, exist_ok=True)
			metrics_list.append(metrics)
		else:
			log_dir = log.output_dir

		cm_path = os.path.join(log_dir, "confusion_matrix.png")
		roc_path = os.path.join(log_dir, "roc_curve.png")
		save_confusion_matrix(cm, cm_path)
		if not args.fixed_specificity:
			save_roc_curve(fpr, tpr, roc_auc, roc_path)

		metrics_filename = "test_metrics.json"
		if args.fixed_specificity:
			metrics_filename = f"test_metrics{fixed_specificity_suffix(True, args.fixed_specificity_value)}.json"
		metrics_output_path = os.path.join(log_dir, metrics_filename)
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
		if not args.fixed_specificity:
			log.info(f"Saved ROC curve to: {roc_path}")
		log.info(f"Saved predictions to: {preds_output_path}")

	# Aggregate results across folds or seeds if requested
	if args.aggregate_results and num_runs > 1:
		if args.multiple_seeds:
			agg_key = "seed"
		else:
			agg_key = "fold"
		log.info(f"Aggregating metrics across {agg_key}s...")

		# Compute aggregated metrics
		agg_metrics = {}
		for metric_name in metrics_list[0].keys():
			if metric_name in ["dataset", "model_type", "checkpoint_path"]:
				continue  # Skip non-numeric metadata fields
			metric_values = [m[metric_name] for m in metrics_list]
			if all(isinstance(v, (int, float)) for v in metric_values):
				if metric_name == "num_samples":
					agg_metrics[metric_name] = metric_values 
					continue
				agg_metrics[metric_name] = {
					"mean": statistics.mean(metric_values),
					"std": round(statistics.stdev(metric_values), 4) if len(metric_values) > 1 else 0.0,
				}
				log.info(f"{metric_name}: {agg_metrics[metric_name]['mean']:.4f} ± {agg_metrics[metric_name]['std']:.4f}")
			else:
				log.warning(f"Skipping aggregation for non-numeric metric: {metric_name}")

		# Save aggregated metrics
		agg_output_path = os.path.join(
			log.output_dir,
			f"aggregated_{agg_key}_metrics{fixed_specificity_suffix(args.fixed_specificity, args.fixed_specificity_value)}.json",
		)
		with open(agg_output_path, "w") as f:
			json.dump(agg_metrics, f, indent=2)
		log.info(f"Saved aggregated metrics to: {agg_output_path}")



if __name__ == "__main__":
	main()
