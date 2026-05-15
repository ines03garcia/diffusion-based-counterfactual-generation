"""
Aggregate and normalize LOW-score CSVs produced per-seed and per-run.

Usage examples:
	python src/E_Aux_Scripts/low_analyzer.py --run_parent_dir /runs/expA \
		--cf_dir data/images/repaint_results --metadata_path data/metadata/resized_df_512.json

	python src/E_Aux_Scripts/low_analyzer.py --run_parent_dirs p1 p2 p3 p4 \
		--cf_dir data/images/repaint_results
"""

import pandas as pd
import argparse
import os
import sys
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
METADATA_ROOT = os.path.join(PROJECT_ROOT, "data/metadata")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data/images")


def main():
	parser = create_argparser()
	args = parser.parse_args()

	parent_dirs = resolve_parent_dirs(args)
	# Aggregate seeds inside each parent dir
	parent_aggregates = []
	for parent in parent_dirs:
		parent_df = aggregate_seeds_in_parent(parent)
		# save parent aggregate
		parent_agg_path = os.path.join(parent, "low_scores_aggregated_by_seeds.csv")
		parent_df.to_csv(parent_agg_path, index=False)
		parent_aggregates.append(parent_df)

	# If multiple parents (4), aggregate them; otherwise use the single parent aggregate
	if len(parent_aggregates) == 1:
		df = parent_aggregates[0]
		output_path = parent_dirs[0]
	else:
		# merge the parent aggregates by image_id,label and average avg_* columns
		df = aggregate_parent_aggregates(parent_aggregates)
		# save a combined aggregated CSV and JSON in a dedicated folder under the first parent
		output_base = parent_dirs[0]
		combined_dir = os.path.join(output_base, "aggregated_4_parents")
		os.makedirs(combined_dir, exist_ok=True)
		csv_path = os.path.join(combined_dir, "low_scores_multiple_seeds_aggregated.csv")
		df.to_csv(csv_path, index=False)
		json_path = os.path.join(combined_dir, "low_scores_multiple_seeds_aggregated.json")
		with open(json_path, 'w', encoding='utf-8') as jf:
			jf.write(df.to_json(orient='records', indent=2))
		output_path = combined_dir

	ranking_columns = resolve_ranking_columns(df, ["avg_low", "avg_sample_loss", "avg_loss_grad_norm"]) 

	# Sort counterfactuals from highest score to lowest
	top_10_percent = len(df) // 10

	for col in ranking_columns:
		df_sorted = df.sort_values(by=[col], ascending=False)
		df_sorted_limited = df_sorted[:top_10_percent] # For each sorted df, computed the highest 10% of values and save them in a different folder
		
		with open(args.metadata_path, 'r') as f:
			metadata = json.load(f)

		for img in range(len(df_sorted_limited)):
			img_name = df_sorted_limited.iloc[img]["image_id"]
			metadata = [entry for entry in metadata if entry.get("image_id") != img_name]
			img_path = os.path.join(args.cf_dir, img_name)
			output_img_path = os.path.join(output_path, col, img_name)
			os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
			os.system(f"cp {img_path} {output_img_path}")
		
		output_metadata_path = os.path.join(output_path, f"metadata_{col}.json")
		os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
		with open(output_metadata_path, 'w', encoding='utf-8') as f:
			json.dump(metadata, f, indent=2, ensure_ascii=False)
			f.write("\n")

	return 1


def resolve_parent_dirs(args):
	if args.run_parent_dir and args.run_parent_dirs:
		raise ValueError("Provide only one of --run_parent_dir or --run_parent_dirs")
	if args.run_parent_dir:
		if not os.path.isdir(args.run_parent_dir):
			raise ValueError(f"run_parent_dir not found: {args.run_parent_dir}")
		return [args.run_parent_dir]
	if args.run_parent_dirs:
		if len(args.run_parent_dirs) != 4:
			raise ValueError("--run_parent_dirs requires exactly 4 folder paths")
		for d in args.run_parent_dirs:
			if not os.path.isdir(d):
				raise ValueError(f"run_parent_dir not found: {d}")
		return list(args.run_parent_dirs)
	raise ValueError("Provide either --run_parent_dir or --run_parent_dirs")


def aggregate_seeds_in_parent(parent_dir):
	seed_paths = sorted(glob.glob(os.path.join(parent_dir, "seed_*", "low_scores_*.csv")))
	if not seed_paths:
		raise ValueError(f"No seed low-score CSVs found under parent: {parent_dir}")
	# normalize each seed file
	dfs = [normalize_low_scores_frame(pd.read_csv(p)) for p in seed_paths]
	# now aggregate across seeds similar to previous logic
	merge_columns = ["image_id", "label"]
	value_columns = ["avg_low", "avg_sample_loss", "avg_loss_grad_norm", "sum_low", "count", "last_low"]
	merged = dfs[0][merge_columns + value_columns].copy()
	merged = merged.rename(columns={column: f"{column}_0" for column in value_columns})
	for index, frame in enumerate(dfs[1:], start=1):
		frame_to_merge = frame[merge_columns + value_columns].copy().rename(
			columns={column: f"{column}_{index}" for column in value_columns}
		)
		merged = merged.merge(frame_to_merge, on=merge_columns, how="inner")

	for column in ["avg_low", "avg_sample_loss", "avg_loss_grad_norm"]:
		merged[column] = merged[[f"{column}_{index}" for index in range(len(dfs))]].mean(axis=1)

	agg = pd.DataFrame({
		"image_id": merged["image_id"],
		"label": merged["label"],
		"sum_low": merged[[f"sum_low_{index}" for index in range(len(dfs))]].sum(axis=1),
		"count": merged[[f"count_{index}" for index in range(len(dfs))]].sum(axis=1),
		"avg_low": merged["avg_low"],
		"avg_sample_loss": merged["avg_sample_loss"],
		"avg_loss_grad_norm": merged["avg_loss_grad_norm"],
		"last_low": merged[[f"last_low_{index}" for index in range(len(dfs))]].mean(axis=1),
	})
	agg["count"] = agg["count"].astype(int)
	return agg


def aggregate_parent_aggregates(parent_aggregates):
	# parent_aggregates is list of dataframes with columns image_id,label,sum_low,count,avg_low,avg_sample_loss,avg_loss_grad_norm,last_low
	merge_columns = ["image_id", "label"]
	value_columns = ["avg_low", "avg_sample_loss", "avg_loss_grad_norm", "sum_low", "count", "last_low"]
	merged = parent_aggregates[0][merge_columns + value_columns].copy()
	merged = merged.rename(columns={column: f"{column}_0" for column in value_columns})
	for index, frame in enumerate(parent_aggregates[1:], start=1):
		frame_to_merge = frame[merge_columns + value_columns].copy().rename(
			columns={column: f"{column}_{index}" for column in value_columns}
		)
		merged = merged.merge(frame_to_merge, on=merge_columns, how="inner")

	for column in ["avg_low", "avg_sample_loss", "avg_loss_grad_norm"]:
		merged[column] = merged[[f"{column}_{index}" for index in range(len(parent_aggregates))]].mean(axis=1)

	agg = pd.DataFrame({
		"image_id": merged["image_id"],
		"label": merged["label"],
		"sum_low": merged[[f"sum_low_{index}" for index in range(len(parent_aggregates))]].sum(axis=1),
		"count": merged[[f"count_{index}" for index in range(len(parent_aggregates))]].sum(axis=1),
		"avg_low": merged["avg_low"],
		"avg_sample_loss": merged["avg_sample_loss"],
		"avg_loss_grad_norm": merged["avg_loss_grad_norm"],
		"last_low": merged[[f"last_low_{index}" for index in range(len(parent_aggregates))]].mean(axis=1),
	})
	agg["count"] = agg["count"].astype(int)
	return agg


def normalize_low_scores_frame(df):
	if {"avg_low", "avg_sample_loss", "avg_loss_grad_norm", "sum_low", "count", "last_low"}.issubset(df.columns):
		return df[["image_id", "label", "sum_low", "count", "avg_low", "avg_sample_loss", "avg_loss_grad_norm", "last_low"]].copy()

	if {"low_score", "sample_loss", "loss_grad_norm"}.issubset(df.columns):
		normalized = pd.DataFrame({
			"image_id": df["image_id"],
			"label": df["label"],
			"sum_low": df["low_score"],
			"count": 1,
			"avg_low": df["low_score"],
			"avg_sample_loss": df["sample_loss"],
			"avg_loss_grad_norm": df["loss_grad_norm"],
			"last_low": df["low_score"],
		})
		return normalized

	raise ValueError(f"Unsupported LOW-score schema: {list(df.columns)}")


def resolve_ranking_columns(df, preferred_columns):
	if all(column in df.columns for column in preferred_columns):
		return preferred_columns

	fallback_columns = ["low_score", "sample_loss", "loss_grad_norm"]
	if all(column in df.columns for column in fallback_columns):
		return fallback_columns

	available_columns = [
		column for column in [
			"avg_low",
			"avg_sample_loss",
			"avg_loss_grad_norm",
			"low_score",
			"sample_loss",
			"loss_grad_norm",
		] if column in df.columns
	]
	if len(available_columns) >= 3:
		return available_columns[:3]

	raise ValueError(f"Could not find the required LOW score columns in dataframe: {list(df.columns)}")


def resolve_output_path(args, low_score_paths):
	if args.run_parent_dir:
		return args.run_parent_dir
	return os.path.dirname(low_score_paths[0])


def create_argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--run_parent_dir", type=str, default=None)
	parser.add_argument("--run_parent_dirs", nargs=4, type=str, default=None)
	parser.add_argument('--metadata_path', type=str, default=os.path.join(METADATA_ROOT, "resized_df_512.json"))
	parser.add_argument('--cf_dir', type=str, default=os.path.join(IMAGES_ROOT, "repaint_results"))
	return parser

if __name__ == "__main__":
	main()
