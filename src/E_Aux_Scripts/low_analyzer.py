import pandas as pd
import argparse
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
METADATA_ROOT = os.path.join(PROJECT_ROOT, "data/metadata")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data/images")


def main():
	parser = create_argparser()
	args = parser.parse_args()

	df = pd.read_csv(args.low_scores_path)
	columns = ["low_score", "sample_loss", "loss_grad_norm"]

	# Sort counterfactuals from highest score to lowest
	top_10_percent = len(df) // 10
	output_path = os.path.dirname(args.low_scores_path)

	for col in columns:
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


def create_argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--low_scores_path", type=str, default=None)
	parser.add_argument('--metadata_path', type=str, default=os.path.join(METADATA_ROOT, "resized_df_512.json"))
	parser.add_argument('--cf_dir', type=str, default=os.path.join(IMAGES_ROOT, "repaint_results"))
	return parser

if __name__ == "__main__":
	main()
