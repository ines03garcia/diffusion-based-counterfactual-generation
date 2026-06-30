"""Process the single-rater, two-task radiologist assessment."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
METADATA_ROOT = DATA_ROOT / "metadata"
RATINGS = [1, 2, 3, 4, 5]
TASK_LABELS = {
    "task1": "Full Image",
    "task2": "Bounding Box",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Join the two single-rater assessment tasks with image metadata "
            "and generate rating visualizations."
        )
    )
    parser.add_argument(
        "--task1_csv",
        type=Path,
        default=METADATA_ROOT / "new_radiologist_assessment_task1.csv",
    )
    parser.add_argument(
        "--task2_csv",
        type=Path,
        default=METADATA_ROOT / "new_radiologist_assessment_task2.csv",
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=METADATA_ROOT / "radiologist_dataset.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_ROOT / "results/new_radiologist_assessment",
    )
    return parser.parse_args()


def normalize_image_id(value):
    """Use only the filename so CSV paths and metadata IDs join consistently."""
    return Path(str(value)).name


def load_task(csv_path, task_name):
    """Load one task and enforce exactly one valid rating per image."""
    df = pd.read_csv(csv_path)
    required = {"image_name", "score", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    df = df.loc[:, ["image_name", "score", "timestamp"]].copy()
    df["image_id"] = df["image_name"].map(normalize_image_id)
    df["rating"] = pd.to_numeric(df["score"], errors="raise")

    invalid_ratings = df.loc[~df["rating"].isin(RATINGS), "rating"].tolist()
    if invalid_ratings:
        raise ValueError(
            f"{task_name} contains ratings outside 1-5: {invalid_ratings}"
        )

    counts = df.groupby("image_id", dropna=False).size()
    repeated = counts[counts != 1]
    if not repeated.empty:
        raise ValueError(
            f"{task_name} must have exactly one rating per image; found: "
            f"{repeated.to_dict()}"
        )

    df["rating"] = df["rating"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df.drop(columns=["image_name", "score"])


def load_metadata(metadata_path):
    with metadata_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    metadata = pd.DataFrame(records)
    if "image_id" not in metadata:
        raise ValueError(f"Metadata file has no image_id field: {metadata_path}")

    metadata["image_id"] = metadata["image_id"].map(normalize_image_id)
    duplicates = metadata.loc[metadata["image_id"].duplicated(), "image_id"].tolist()
    if duplicates:
        raise ValueError(f"Metadata contains duplicate image IDs: {duplicates}")
    return metadata


def validate_same_subset(task1, task2):
    task1_ids = set(task1["image_id"])
    task2_ids = set(task2["image_id"])
    if task1_ids != task2_ids:
        raise ValueError(
            "Tasks do not contain the same image subset. "
            f"Only in task 1: {sorted(task1_ids - task2_ids)}; "
            f"only in task 2: {sorted(task2_ids - task1_ids)}"
        )


def join_metadata(task, metadata, task_name):
    joined = task.merge(
        metadata,
        on="image_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = joined.loc[joined["_merge"] != "both", "image_id"].tolist()
    if unmatched:
        raise ValueError(f"Metadata is missing {task_name} images: {unmatched}")
    return joined.drop(columns="_merge")


def plot_alternative_rating_visualizations(task1, task2, output_dir):
    """Create the grouped four-bar view of healthy and counterfactual ratings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    percentages = {}
    for task_name, task_df in (("task1", task1), ("task2", task2)):
        for has_cf in (0, 1):
            subset = task_df.loc[task_df["has_cf"] == has_cf]
            counts = subset["rating"].value_counts().reindex(RATINGS, fill_value=0)
            percentages[(task_name, has_cf)] = counts.div(len(subset)).mul(100).to_numpy()

    colors = {0: "#6FA8DC", 1: "#E69138"}
    hatches = {"task1": "", "task2": "///"}
    task_labels = {"task1": "Full Image", "task2": "Bounding Box"}
    type_labels = {0: "Real Healthy", 1: "Counterfactual Healthy"}
    series_order = (("task1", 0), ("task1", 1), ("task2", 0), ("task2", 1))

    fig, ax = plt.subplots(figsize=(14, 6))
    x_positions = np.arange(len(RATINGS))
    bar_width = 0.16
    bar_spacing = bar_width
    for offset_index, (task_name, has_cf) in enumerate(series_order):
        offset = (offset_index - 1.5) * bar_spacing
        ax.bar(x_positions + offset, percentages[(task_name, has_cf)], width=bar_width, color=colors[has_cf], edgecolor="black", hatch=hatches[task_name])
    ax.set_title("Real Healthy vs Counterfactual Healthy Rating Comparison", fontsize=19)
    ax.set_xlabel("Rating", fontsize=16)
    ax.set_ylabel("Percentage of images (%)", fontsize=16)
    ax.set_xticks(x_positions, RATINGS)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    type_legend = ax.legend(handles=[Patch(facecolor=colors[0], edgecolor="black", label=type_labels[0]), Patch(facecolor=colors[1], edgecolor="black", label=type_labels[1])], title="Dataset", loc="upper left", fontsize=14, title_fontsize=14)
    ax.add_artist(type_legend)
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", label=task_labels["task1"]), Patch(facecolor="white", edgecolor="black", hatch=hatches["task2"], label=task_labels["task2"])], title="Task", loc="upper right", fontsize=14, title_fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "01_four_bars_color_type_hatch_task.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_paired_rating_matrix(df, output_path, title):
    matrix = pd.crosstab(
        df["task1_rating"], df["task2_rating"]
    ).reindex(index=RATINGS, columns=RATINGS, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix.values, cmap="Blues", origin="lower")
    for row in range(len(RATINGS)):
        for column in range(len(RATINGS)):
            ax.text(column, row, matrix.iloc[row, column], ha="center", va="center")
    ax.set_xticks(range(len(RATINGS)), RATINGS)
    ax.set_yticks(range(len(RATINGS)), RATINGS)
    ax.set_xlabel("Task 2 Rating (Bounding Box)")
    ax.set_ylabel("Task 1 Rating (Full Image)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Number of Images")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task_comparison(combined, output_dir):
    plot_paired_rating_matrix(
        combined,
        output_dir / "paired_rating_matrix.png",
        f"Paired Ratings Across Tasks – All Images (n={len(combined)})",
    )
    plot_paired_rating_matrix(
        combined.loc[combined["has_cf"] == 1],
        output_dir / "paired_rating_matrix_counterfactual.png",
        f"Paired Ratings Across Tasks – Counterfactuals "
        f"(n={(combined['has_cf'] == 1).sum()})",
    )
    plot_paired_rating_matrix(
        combined.loc[combined["has_cf"] == 0],
        output_dir / "paired_rating_matrix_healthy.png",
        f"Paired Ratings Across Tasks – Healthy Images "
        f"(n={(combined['has_cf'] == 0).sum()})",
    )


def build_combined_table(task1, task2):
    task2_ratings = task2.loc[:, ["image_id", "rating", "timestamp"]].rename(
        columns={"rating": "task2_rating", "timestamp": "task2_timestamp"}
    )
    combined = task1.rename(
        columns={"rating": "task1_rating", "timestamp": "task1_timestamp"}
    )
    return combined.merge(task2_ratings, on="image_id", validate="one_to_one")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task1_raw = load_task(args.task1_csv, "task 1")
    task2_raw = load_task(args.task2_csv, "task 2")
    validate_same_subset(task1_raw, task2_raw)

    metadata = load_metadata(args.metadata_path)
    task1 = join_metadata(task1_raw, metadata, "task 1")
    task2 = join_metadata(task2_raw, metadata, "task 2")
    combined = build_combined_table(task1, task2)

    plot_alternative_rating_visualizations(
        task1,
        task2,
        args.output_dir / "rating_visualization_alternatives",
    )
    plot_task_comparison(combined, args.output_dir)

    print("Verified exactly one rating per image in each task.")
    print(f"Verified the same {len(task1)} images are present in both tasks.")
    print(f"Joined all ratings with metadata from {args.metadata_path}.")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
