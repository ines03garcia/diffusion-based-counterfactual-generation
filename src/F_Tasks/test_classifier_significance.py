import argparse
import importlib
import json
import os
import sys
from typing import List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.E_Aux_Scripts.argument_parsers import create_significance_argparser

LOWER_IS_BETTER_METRICS = {"log_loss"}


def is_metrics_file_name(file_name: str) -> bool:
	return (
		file_name == "test_metrics.json"
		or (
			file_name.startswith("test_metrics_fixed_specificity_")
			and file_name.endswith(".json")
		)
	)


def find_metrics_files(root_path: str) -> List[str]:
	if os.path.isfile(root_path):
		return [root_path]

	metrics_files = []
	for current_root, _, files in os.walk(root_path):
		for file_name in files:
			if is_metrics_file_name(file_name):
				metrics_files.append(os.path.join(current_root, file_name))

	return sorted(metrics_files)


def load_metric_values(root_path: str, metric_name: str) -> List[float]:
	metrics_files = find_metrics_files(root_path)
	if not metrics_files:
		raise FileNotFoundError(f"No test metrics JSON files found under: {root_path}")

	metric_values = []
	for metrics_file in metrics_files:
		with open(metrics_file, "r") as f:
			metrics = json.load(f)

		value = metrics.get(metric_name)
		if isinstance(value, (int, float)):
			metric_values.append(float(value))

	if not metric_values:
		raise ValueError(
			f"No numeric '{metric_name}' values found under: {root_path}. "
			"Use a scalar run-level metric from a test metrics JSON file, not predictions.json."
		)

	return metric_values


def compare_with_aso(values_baseline: List[float], values_cf: List[float], seed: int, lower_is_better: bool):
	try:
		aso = importlib.import_module("deepsig").aso
	except ImportError as exc:
		raise ImportError("The 'deepsig' package is required. Install it with pip install deepsig.") from exc

	comparison_baseline = np.asarray(values_baseline, dtype=float)
	comparison_cf = np.asarray(values_cf, dtype=float)

	if lower_is_better:
		comparison_baseline = -comparison_baseline
		comparison_cf = -comparison_cf

	min_eps = float(aso(comparison_baseline, comparison_cf, seed=seed))
	mean_baseline = float(np.mean(values_baseline))
	mean_cf = float(np.mean(values_cf))
	mean_gap = mean_baseline - mean_cf
	oriented_gap = float(np.mean(comparison_baseline) - np.mean(comparison_cf))

	result = {
		"values_baseline": values_baseline,
		"values_cf": values_cf,
		"mean_baseline": mean_baseline,
		"mean_cf": mean_cf,
		"mean_gap_baseline_minus_cf": mean_gap,
		"mean_gap_oriented": oriented_gap,
		"aso_min_eps": min_eps,
		"lower_is_better": bool(lower_is_better),
		"baseline_better_by_aso": bool(oriented_gap > min_eps),
		"cf_better_by_aso": bool((-oriented_gap) > min_eps),
	}
	return result


def parse_args():
	parser = create_significance_argparser()
	return parser.parse_args()


def resolve_output_path(baseline_aug: str, cf_aug: str, output_path: str):
	baseline_name = os.path.basename(os.path.normpath(baseline_aug))
	cf_name = os.path.basename(os.path.normpath(cf_aug))
	default_file_name = f"significance_{baseline_name}_vs_{cf_name}.json"

	if output_path is None:
		parent_dir = os.path.dirname(os.path.normpath(os.path.abspath(baseline_aug)))
		return os.path.join(parent_dir, default_file_name)

	# If the user passes a directory path, save with the default file name in that directory.
	if os.path.isdir(output_path):
		return os.path.join(output_path, default_file_name)

	return output_path


def round_floats(obj, ndigits=2):
	if isinstance(obj, float):
		return round(obj, ndigits)
	if isinstance(obj, list):
		return [round_floats(item, ndigits=ndigits) for item in obj]
	if isinstance(obj, dict):
		return {key: round_floats(value, ndigits=ndigits) for key, value in obj.items()}
	return obj


def main():
	args = parse_args()

	if not os.path.exists(args.baseline_aug):
		raise FileNotFoundError(f"Path not found: {args.baseline_aug}")
	if not os.path.exists(args.cf_aug):
		raise FileNotFoundError(f"Path not found: {args.cf_aug}")

	result = {
		"baseline_aug": args.baseline_aug,
		"cf_aug": args.cf_aug,
		"seed": args.seed,
		"metrics": {},
	}

	for metric_name in args.metrics:
		metric_values_baseline = load_metric_values(args.baseline_aug, metric_name)
		metric_values_cf = load_metric_values(args.cf_aug, metric_name)
		if len(metric_values_baseline) != len(metric_values_cf):
			print(
				f"Warning for {metric_name}: comparing {len(metric_values_baseline)} baseline values against {len(metric_values_cf)} cf values."
			)

		metric_result = compare_with_aso(
			metric_values_baseline,
			metric_values_cf,
			seed=args.seed,
			lower_is_better=(metric_name in LOWER_IS_BETTER_METRICS),
		)
		metric_result.update({"metric": metric_name})
		result["metrics"][metric_name] = metric_result

	result = round_floats(result, ndigits=2)

	print(json.dumps(result, indent=2))

	output_file_path = resolve_output_path(args.baseline_aug, args.cf_aug, args.output_path)
	os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
	with open(output_file_path, "w") as f:
		json.dump(result, f, indent=2)

	print(f"Saved significance results to: {output_file_path}")

	baseline_better_metrics = []
	cf_better_metrics = []
	inconclusive_metrics = []

	for metric_name, metric_data in result["metrics"].items():
		baseline_better = bool(metric_data.get("baseline_better_by_aso", False))
		cf_better = bool(metric_data.get("cf_better_by_aso", False))

		if baseline_better and not cf_better:
			baseline_better_metrics.append(metric_name)
		elif cf_better and not baseline_better:
			cf_better_metrics.append(metric_name)
		else:
			inconclusive_metrics.append(metric_name)

	comparison_summary = {
		"baseline_aug": result["baseline_aug"],
		"cf_aug": result["cf_aug"],
		"seed": result["seed"],
		"baseline_better_metrics": baseline_better_metrics,
		"cf_better_metrics": cf_better_metrics,
		"inconclusive_metrics": inconclusive_metrics,
	}

	base_path, extension = os.path.splitext(output_file_path)
	if extension == "":
		extension = ".json"
	summary_output_path = f"{base_path}_comparison_summary{extension}"

	with open(summary_output_path, "w") as f:
		json.dump(comparison_summary, f, indent=2)

	print(f"Saved comparison summary to: {summary_output_path}")


if __name__ == "__main__":
	main()