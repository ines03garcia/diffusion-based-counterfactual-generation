"""Process the single-rater, two-task radiologist assessment."""

import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def plot_rating_distributions(
    task1,
    task2,
    output_path,
    subset_label=None,
    as_percentages=False,
    bar_color="#6FA8DC",
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, df, task_name in zip(axes, [task1, task2], TASK_LABELS):
        counts = df["rating"].value_counts().reindex(RATINGS, fill_value=0)
        values = counts.div(len(df)).mul(100) if as_percentages else counts
        ax.bar(RATINGS, values.values, color=bar_color, edgecolor="black")
        plot_type = "Rating Percentages" if as_percentages else "Rating Distribution"
        ax.set_title(f"{plot_type} – {TASK_LABELS[task_name]}")
        ax.set_xlabel("Rating")
        ax.set_ylabel(
            "Percentage of Images (%)" if as_percentages else "Number of Images"
        )
        ax.set_xticks(RATINGS)
        if as_percentages:
            ax.set_ylim(0, 80)
            ax.set_yticks(np.arange(0, 81, 10))
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.5)
    if subset_label:
        plot_type = "Rating Percentages" if as_percentages else "Rating Distributions"
        fig.suptitle(
            f"{plot_type} – {subset_label} (n={len(task1)})",
            fontsize=14,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rating_distributions_by_cf_status(
    task1, task2, output_path, as_percentages=False
):
    """Plot healthy and counterfactual rating bars together for both tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x_positions = np.arange(len(RATINGS))
    bar_width = 0.38
    groups = [
        (0, "Healthy", "#6FA8DC", -bar_width / 2),
        (1, "Counterfactual", "#E69138", bar_width / 2),
    ]

    for ax, df, task_name in zip(axes, [task1, task2], TASK_LABELS):
        for has_cf, label, color, offset in groups:
            subset = df.loc[df["has_cf"] == has_cf]
            counts = subset["rating"].value_counts().reindex(RATINGS, fill_value=0)
            values = counts.div(len(subset)).mul(100) if as_percentages else counts
            ax.bar(
                x_positions + offset,
                values.values,
                width=bar_width,
                color=color,
                edgecolor="black",
                label=label,
            )

        ax.set_title(f"Healthy vs Counterfactual – {TASK_LABELS[task_name]}")
        ax.set_xlabel("Rating")
        ax.set_ylabel(
            "Percentage Within Group (%)" if as_percentages else "Number of Images"
        )
        ax.set_xticks(x_positions, RATINGS)
        ax.legend()
        if as_percentages:
            ax.set_ylim(0, 80)
            ax.set_yticks(np.arange(0, 81, 10))
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.5)

    plot_type = "Percentage" if as_percentages else "Count"
    fig.suptitle(f"Rating Distributions by Image Type ({plot_type})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ratings_by_group(df, group_col, task_name, output_path, group_labels=None):
    plot_df = df.loc[df[group_col].notna(), [group_col, "rating"]].copy()
    if group_labels:
        plot_df[group_col] = plot_df[group_col].map(group_labels).fillna(
            plot_df[group_col].astype(str)
        )

    percentages = (
        pd.crosstab(plot_df[group_col], plot_df["rating"], normalize="index")
        .reindex(columns=RATINGS, fill_value=0)
        .mul(100)
        .T
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    percentages.plot(kind="bar", ax=ax, width=0.8, edgecolor="white")
    ax.set_title(
        f"Rating Distribution by {group_col.replace('_', ' ').title()} "
        f"– {TASK_LABELS[task_name]}"
    )
    ax.set_xlabel("Rating")
    ax.set_ylabel("Percentage Within Group (%)")
    ax.set_xticklabels(RATINGS, rotation=0)
    ax.legend(title=group_col.replace("_", " ").title())
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 10))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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

    differences = combined["task2_rating"] - combined["task1_rating"]
    difference_values = list(range(-4, 5))
    counts = differences.value_counts().reindex(difference_values, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(difference_values, counts.values, color="#93C47D", edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(difference_values)
    ax.set_xlabel("Task 2 Rating − Task 1 Rating")
    ax.set_ylabel("Number of Images")
    ax.set_title("Paired Rating Differences")
    fig.tight_layout()
    fig.savefig(output_dir / "paired_rating_differences.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_coordinate_list(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except (ValueError, SyntaxError):
            return []
    return [value]


def annotation_area(row):
    coordinates = [
        parse_coordinate_list(row.get(column))
        for column in ["resized_xmin", "resized_ymin", "resized_xmax", "resized_ymax"]
    ]
    area = 0.0
    for xmin, ymin, xmax, ymax in zip(*coordinates):
        width = float(xmax) - float(xmin)
        height = float(ymax) - float(ymin)
        if width > 0 and height > 0:
            area += width * height
    return area if area > 0 else np.nan


def plot_annotation_area_vs_rating(task2, output_path):
    plot_df = task2.copy()
    plot_df["annotation_area"] = plot_df.apply(annotation_area, axis=1) / 10000.0
    plot_df = plot_df.dropna(subset=["annotation_area", "rating"])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        plot_df["annotation_area"],
        plot_df["rating"],
        alpha=0.65,
        s=55,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Annotation Area (×10⁴ pixels²)")
    ax.set_ylabel("Task 2 Rating")
    ax.set_yticks(RATINGS)
    ax.set_title("Annotation Area vs Bounding-Box Rating")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_combined_table(task1, task2):
    task2_ratings = task2.loc[:, ["image_id", "rating", "timestamp"]].rename(
        columns={"rating": "task2_rating", "timestamp": "task2_timestamp"}
    )
    combined = task1.rename(
        columns={"rating": "task1_rating", "timestamp": "task1_timestamp"}
    )
    return combined.merge(task2_ratings, on="image_id", validate="one_to_one")


def save_verification_summary(task1, task2, combined, output_path):
    summary = {
        "same_image_subset": set(task1["image_id"]) == set(task2["image_id"]),
        "task1": {
            "number_of_images": int(len(task1)),
            "maximum_ratings_per_image": int(task1.groupby("image_id").size().max()),
            "rating_counts": {
                str(key): int(value)
                for key, value in task1["rating"].value_counts().sort_index().items()
            },
        },
        "task2": {
            "number_of_images": int(len(task2)),
            "maximum_ratings_per_image": int(task2.groupby("image_id").size().max()),
            "rating_counts": {
                str(key): int(value)
                for key, value in task2["rating"].value_counts().sort_index().items()
            },
        },
        "metadata_columns": [
            column
            for column in combined.columns
            if column not in {"task1_rating", "task2_rating", "task1_timestamp", "task2_timestamp"}
        ],
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


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

    task1.to_csv(args.output_dir / "task1_with_metadata.csv", index=False)
    task2.to_csv(args.output_dir / "task2_with_metadata.csv", index=False)
    combined.to_csv(args.output_dir / "combined_tasks_with_metadata.csv", index=False)
    save_verification_summary(
        task1,
        task2,
        combined,
        args.output_dir / "verification_summary.json",
    )

    plot_rating_distributions(
        task1, task2, args.output_dir / "rating_distributions.png"
    )
    plot_rating_distributions(
        task1.loc[task1["has_cf"] == 1],
        task2.loc[task2["has_cf"] == 1],
        args.output_dir / "rating_distributions_counterfactual.png",
        subset_label="Counterfactual Images",
        bar_color="#E69138",
    )
    plot_rating_distributions(
        task1.loc[task1["has_cf"] == 0],
        task2.loc[task2["has_cf"] == 0],
        args.output_dir / "rating_distributions_healthy.png",
        subset_label="Healthy Images",
    )
    plot_rating_distributions(
        task1,
        task2,
        args.output_dir / "rating_percentages.png",
        as_percentages=True,
    )
    plot_rating_distributions(
        task1.loc[task1["has_cf"] == 1],
        task2.loc[task2["has_cf"] == 1],
        args.output_dir / "rating_percentages_counterfactual.png",
        subset_label="Counterfactual Images",
        as_percentages=True,
        bar_color="#E69138",
    )
    plot_rating_distributions(
        task1.loc[task1["has_cf"] == 0],
        task2.loc[task2["has_cf"] == 0],
        args.output_dir / "rating_percentages_healthy.png",
        subset_label="Healthy Images",
        as_percentages=True,
    )
    plot_rating_distributions_by_cf_status(
        task1,
        task2,
        args.output_dir / "rating_distributions_healthy_vs_counterfactual.png",
    )
    plot_rating_distributions_by_cf_status(
        task1,
        task2,
        args.output_dir / "rating_percentages_healthy_vs_counterfactual.png",
        as_percentages=True,
    )

    group_specs = [
        ("breast_density", None),
        ("breast_birads", None),
        ("has_cf", {0: "Healthy", 1: "Counterfactual"}),
    ]
    for task_name, task_df in [("task1", task1), ("task2", task2)]:
        for group_column, labels in group_specs:
            if group_column in task_df:
                plot_ratings_by_group(
                    task_df,
                    group_column,
                    task_name,
                    args.output_dir / f"ratings_by_{group_column}_{task_name}.png",
                    group_labels=labels,
                )

    plot_task_comparison(combined, args.output_dir)
    plot_annotation_area_vs_rating(
        task2, args.output_dir / "annotation_area_vs_task2_rating.png"
    )

    print("Verified exactly one rating per image in each task.")
    print(f"Verified the same {len(task1)} images are present in both tasks.")
    print(f"Joined all ratings with metadata from {args.metadata_path}.")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
