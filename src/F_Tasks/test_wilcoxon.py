import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
METADATA_ROOT = os.path.join(PROJECT_ROOT, "data/metadata")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data/images")

from src.E_Aux_Scripts import logger
from src.E_Aux_Scripts.argument_parsers import create_test_argparser
from src.E_Aux_Scripts.classifier_helpers import (
	build_model,
	create_transforms,
	get_test_dataset,
	load_checkpoint_weights,
	run_inference,
)
from src.E_Aux_Scripts.utils import seed_worker, set_seed


def create_wilcoxon_argparser():
	parser = create_test_argparser()
	parser.description = (
		"Compare two classifier checkpoints with a paired Wilcoxon signed-rank "
		"test over per-image true-class probabilities."
	)
	parser.add_argument(
		"--baseline_checkpoint_path",
		required=True,
		help="Path to the baseline augmentation checkpoint. Relative paths are resolved under models/.",
	)
	parser.add_argument(
		"--cf_checkpoint_path",
		required=True,
		help="Path to the counterfactual augmentation checkpoint. Relative paths are resolved under models/.",
	)
	parser.add_argument(
		"--wilcoxon_output_path",
		default=None,
		help="Optional JSON file for the Wilcoxon comparison. Defaults to the logger output directory.",
	)
	return parser


def resolve_checkpoint_path(checkpoint_path):
	if os.path.isabs(checkpoint_path):
		return checkpoint_path
	return os.path.join(MODELS_ROOT, checkpoint_path)


def ensure_default_data_paths(args):
	if args.data_dir is None:
		if args.dataset == "vindr":
			args.data_dir = os.path.join(IMAGES_ROOT, "VinDrMammo-CLIP-CLAHE")
		else:
			args.data_dir = os.path.join(IMAGES_ROOT, "Inbreast_png")

	if args.metadata_path is None:
		if args.dataset == "vindr":
			args.metadata_path = os.path.join(METADATA_ROOT, "processed_df_birads.json")
		else:
			args.metadata_path = os.path.join(METADATA_ROOT, "inbreast_test_metadata.csv")


def infer_true_class_probabilities(model, checkpoint_path, test_loader, device, threshold):
	load_checkpoint_weights(model, checkpoint_path, device)
	probs, _, targets, image_ids = run_inference(model, test_loader, device, threshold=threshold)

	true_class_probs = np.where(targets == 1, probs, 1.0 - probs).astype(np.float64)
	return {
		str(image_id): {
			"target": int(target),
			"prob_class_1": float(prob),
			"true_class_probability": float(true_class_prob),
		}
		for image_id, target, prob, true_class_prob in zip(image_ids, targets, probs, true_class_probs)
	}


def pair_probabilities(baseline_results, cf_results):
	baseline_ids = set(baseline_results)
	cf_ids = set(cf_results)
	if baseline_ids != cf_ids:
		missing_from_cf = sorted(baseline_ids - cf_ids)
		missing_from_baseline = sorted(cf_ids - baseline_ids)
		raise ValueError(
			"Checkpoint outputs do not contain the same image IDs. "
			f"Missing from counterfactual: {missing_from_cf[:10]}; "
			f"missing from baseline: {missing_from_baseline[:10]}"
		)

	image_ids = sorted(baseline_ids)
	for image_id in image_ids:
		baseline_target = baseline_results[image_id]["target"]
		cf_target = cf_results[image_id]["target"]
		if baseline_target != cf_target:
			raise ValueError(
				f"Target mismatch for image_id={image_id}: "
				f"baseline={baseline_target}, counterfactual={cf_target}"
			)

	baseline_probs = np.asarray(
		[baseline_results[image_id]["true_class_probability"] for image_id in image_ids],
		dtype=np.float64,
	)
	cf_probs = np.asarray(
		[cf_results[image_id]["true_class_probability"] for image_id in image_ids],
		dtype=np.float64,
	)
	return image_ids, baseline_probs, cf_probs


def run_wilcoxon(baseline_probs, cf_probs, alternative="greater"):
	differences = cf_probs - baseline_probs
	nonzero_differences = differences[differences != 0.0]

	if len(nonzero_differences) == 0:
		return {
			"statistic": 0.0,
			"p_value": 1.0,
			"alternative": alternative,
			"note": "All paired differences are zero; Wilcoxon statistic is undefined, so p_value was set to 1.0.",
		}

	statistic, p_value = wilcoxon(
		cf_probs,
		baseline_probs,
		alternative=alternative,
		zero_method="wilcox",
	)
	return {
		"statistic": float(statistic),
		"p_value": float(p_value),
		"alternative": alternative,
	}


def summarize_paired_comparison(baseline_probs, cf_probs, alternative="greater"):
	if len(baseline_probs) == 0:
		return {
			"num_pairs": 0,
			"wilcoxon": None,
			"note": "No samples available for this subset.",
		}

	differences = cf_probs - baseline_probs
	wilcoxon_result = run_wilcoxon(baseline_probs, cf_probs, alternative=alternative)
	return {
		"num_pairs": int(len(differences)),
		"wilcoxon": wilcoxon_result,
		"mean_baseline_probability": float(np.mean(baseline_probs)),
		"mean_cf_probability": float(np.mean(cf_probs)),
		"mean_difference": float(np.mean(differences)),
		"median_difference": float(np.median(differences)),
		"num_positive_differences": int(np.sum(differences > 0.0)),
		"num_negative_differences": int(np.sum(differences < 0.0)),
		"num_zero_differences": int(np.sum(differences == 0.0)),
	}


def main():
	parser = create_wilcoxon_argparser()
	args = parser.parse_args()

	if args.output_path is not None:
		raise ValueError("--output_path aggregation is not supported by test_wilcoxon.py.")
	if args.cross_validation or args.multiple_seeds:
		raise ValueError("test_wilcoxon.py compares exactly two checkpoints; cross-validation and multiple-seed modes are not supported.")

	ensure_default_data_paths(args)
	set_seed(args.seed)

	baseline_checkpoint_path = resolve_checkpoint_path(args.baseline_checkpoint_path)
	cf_checkpoint_path = resolve_checkpoint_path(args.cf_checkpoint_path)

	log = logger.Logger(
		experiment_type="Classifiers",
		sub_experiment_type="wilcoxon",
		dataset=args.dataset,
		model_type=args.model_type,
		setup="/baseline_vs_cf_aug",
	)
	log.configure_root_logger()
	log.info(f"Logs will be saved to: {log.output_dir}")

	args_output_path = os.path.join(log.output_dir, "wilcoxon_args.json")
	with open(args_output_path, "w") as f:
		json.dump(vars(args), f, indent=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	log.info(f"Using device [{device}] for Wilcoxon comparison on dataset [{args.dataset}]")

	_, test_transform = create_transforms(
		augmentation_type="none",
		model_type=args.model_type,
	)
	test_dataset = get_test_dataset(args, test_transform)
	log.info(f"Loaded {len(test_dataset)} test samples with class distribution {test_dataset.get_class_distribution()}")

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

	log.info(f"Running baseline checkpoint: {baseline_checkpoint_path}")
	baseline_results = infer_true_class_probabilities(
		model,
		baseline_checkpoint_path,
		test_loader,
		device,
		args.threshold,
	)

	log.info(f"Running counterfactual checkpoint: {cf_checkpoint_path}")
	cf_results = infer_true_class_probabilities(
		model,
		cf_checkpoint_path,
		test_loader,
		device,
		args.threshold,
	)

	image_ids, baseline_probs, cf_probs = pair_probabilities(baseline_results, cf_results)
	targets = np.asarray([baseline_results[image_id]["target"] for image_id in image_ids], dtype=np.int32)
	positive_mask = targets == 1
	negative_mask = targets == 0
	differences = cf_probs - baseline_probs
	wilcoxon_result = run_wilcoxon(baseline_probs, cf_probs)
	overall_summary = summarize_paired_comparison(baseline_probs, cf_probs)
	positive_summary = summarize_paired_comparison(baseline_probs[positive_mask], cf_probs[positive_mask])
	negative_summary = summarize_paired_comparison(baseline_probs[negative_mask], cf_probs[negative_mask])

	result = {
		"dataset": args.dataset,
		"model_type": args.model_type,
		"baseline_checkpoint_path": baseline_checkpoint_path,
		"cf_checkpoint_path": cf_checkpoint_path,
		"num_pairs": int(len(image_ids)),
		"test": "paired Wilcoxon signed-rank",
		"null_hypothesis": "Median paired difference in true-class probability is not greater than zero.",
		"difference": "counterfactual_true_class_probability - baseline_true_class_probability",
		"alternative": "counterfactual checkpoint assigns higher true-class probability than baseline",
		"wilcoxon": wilcoxon_result,
		"mean_baseline_true_class_probability": float(np.mean(baseline_probs)),
		"mean_cf_true_class_probability": float(np.mean(cf_probs)),
		"mean_difference": float(np.mean(differences)),
		"median_difference": float(np.median(differences)),
		"num_positive_differences": int(np.sum(differences > 0.0)),
		"num_negative_differences": int(np.sum(differences < 0.0)),
		"num_zero_differences": int(np.sum(differences == 0.0)),
		"classwise_comparisons": {
			"overall_true_class_probability": overall_summary,
			"class_1_positive_cases": {
				"description": "Positive test images only; probability is P(class 1), so this tests whether CF is better at assigning positives as positive.",
				**positive_summary,
			},
			"class_0_negative_cases": {
				"description": "Negative test images only; probability is P(class 0) = 1 - P(class 1), so this tests whether CF is better at assigning negatives as negative.",
				**negative_summary,
			},
		},
		"paired_probabilities": [
			{
				"image_id": image_id,
				"target": baseline_results[image_id]["target"],
				"baseline_true_class_probability": float(baseline_prob),
				"cf_true_class_probability": float(cf_prob),
				"difference": float(diff),
			}
			for image_id, baseline_prob, cf_prob, diff in zip(image_ids, baseline_probs, cf_probs, differences)
		],
	}

	output_path = args.wilcoxon_output_path
	if output_path is None:
		output_path = os.path.join(log.output_dir, "wilcoxon_results.json")
	elif not os.path.isabs(output_path):
		output_path = os.path.join(PROJECT_ROOT, output_path)

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w") as f:
		json.dump(result, f, indent=2)

	log.info(f"Wilcoxon comparison saved to: {output_path}")
	log.info(
		"Wilcoxon result: "
		f"statistic={wilcoxon_result['statistic']:.6f}, "
		f"p_value={wilcoxon_result['p_value']:.6g}, "
		f"median_difference={result['median_difference']:.6f}"
	)


if __name__ == "__main__":
	main()
