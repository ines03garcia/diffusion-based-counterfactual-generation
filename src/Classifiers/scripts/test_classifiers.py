import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DATASET_DIR, IMAGES_ROOT, METADATA_ROOT
from src.Classifiers.aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset
from src.Classifiers.aux_scripts.Inbreast_dataset import Inbreast_dataset
from src.Classifiers.aux_scripts.ClassifierVisionTransformer import VisionTransformerClassifier
from src.Classifiers.aux_scripts.ClassifierConvNeXt import ConvNeXtClassifier
from src.Classifiers.aux_scripts.utils import create_transforms


def load_model(args, device):
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    model_type = args.model_type.lower()
    if model_type == "vit":
        model = VisionTransformerClassifier(num_classes=1, pretrained=False)
    elif model_type == "convnext":
        model = ConvNeXtClassifier(num_classes=1, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def build_test_dataset(args):
    _, test_transform = create_transforms("none")

    if args.dataset == "vindr":
        data_dir = args.data_dir or DATASET_DIR
        metadata_path = args.metadata_path or str(Path(METADATA_ROOT) / "resized_df_has_counterfactual.csv")
        dataset = VinDrMammo_dataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            split=args.split,
            transform=test_transform,
            use_counterfactuals=False,
        )
    else:
        data_dir = args.data_dir or str(Path(IMAGES_ROOT) / "Inbreast_png")
        metadata_path = args.metadata_path or str(Path(METADATA_ROOT) / "inbreast_test_metadata.csv")
        dataset = Inbreast_dataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            split=args.split,
            transform=test_transform,
        )

    return dataset, data_dir, metadata_path


def run_inference(model, loader, device, threshold):
    all_probs = []
    all_targets = []
    all_image_names = []

    with torch.no_grad():
        for images, labels, names in tqdm(loader, desc="Inference"):
            images = images.to(device)
            labels = labels.cpu().numpy().astype(int)

            logits = model(images)
            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_targets.extend(labels.tolist())
            all_image_names.extend(names)

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(int)

    return y_true, y_prob, y_pred, all_image_names


def compute_metrics(y_true, y_prob, threshold):
    def to_pct(value):
        return round(float(value) * 100.0, 1)

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    y_prob_clipped = np.clip(y_prob, 1e-7, 1.0 - 1e-7)

    metrics = {
        "n_samples": int(len(y_true)),
        "threshold": float(threshold),
        "Acc": to_pct(accuracy_score(y_true, y_pred)),
        "Bacc": to_pct(balanced_accuracy_score(y_true, y_pred)),
        "Prec": to_pct(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": to_pct(recall_score(y_true, y_pred, zero_division=0)),
        "F1": to_pct(f1_score(y_true, y_pred, zero_division=0)),
        "Spec": to_pct(specificity),
        "log_loss": round(float(log_loss(y_true, y_prob_clipped, labels=[0, 1])), 3),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 3),
        "confusion_matrix": cm.tolist(),
    }

    unique = np.unique(y_true)
    if len(unique) > 1:
        metrics["AUC"] = to_pct(roc_auc_score(y_true, y_prob))
        metrics["PR_auc"] = to_pct(average_precision_score(y_true, y_prob))
    else:
        metrics["AUC"] = None
        metrics["PR_auc"] = None

    return metrics, y_pred


def build_results_df(image_names, y_true, y_prob, y_pred):
    results_df = pd.DataFrame(
        {
            "image_name": image_names,
            "true_label": y_true,
            "pred_prob_anomalous": y_prob,
            "pred_label": y_pred,
            "pred_class": np.where(y_pred == 1, "anomalous", "healthy"),
            "correct": (y_pred == y_true).astype(int),
        }
    )

    results_df["confidence"] = np.where(
        results_df["pred_label"] == 1,
        results_df["pred_prob_anomalous"],
        1.0 - results_df["pred_prob_anomalous"],
    )

    return results_df


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Anomalous"],
        yticklabels=["Healthy", "Anomalous"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    unique = np.unique(y_true)
    if len(unique) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_probability_distribution(y_prob, y_true, threshold, save_path):
    healthy_probs = y_prob[y_true == 0]
    anomalous_probs = y_prob[y_true == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(healthy_probs, bins=50, alpha=0.7, label="Healthy (True Label)", color="blue", density=True)
    plt.hist(anomalous_probs, bins=50, alpha=0.7, label="Anomalous (True Label)", color="red", density=True)
    plt.axvline(x=threshold, color="black", linestyle="--", label=f"Decision Threshold ({threshold})")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Probability Distribution by True Class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_outputs(args, metrics, results_df, y_true, y_prob):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    run_folder_name = datetime.now().strftime(f"test_{run_timestamp}")
    run_output_dir = output_dir / run_folder_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    metrics["run_timestamp"] = run_timestamp

    per_image_path = run_output_dir / "inference_per_image.csv"
    args_path = run_output_dir / "test_args.json"
    detailed_results_path = run_output_dir / "detailed_results.csv"
    legacy_metrics_path = run_output_dir / f"test_metrics.json"
    confusion_matrix_path = run_output_dir / "confusion_matrix.png"
    roc_curve_path = run_output_dir / "roc_curve.png"
    probability_distribution_path = run_output_dir / "probability_distribution.png"

    results_df.to_csv(per_image_path, index=False)
    results_df.to_csv(detailed_results_path, index=False)
    with open(legacy_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, confusion_matrix_path)
    plot_roc_curve(y_true, y_prob, roc_curve_path)
    plot_probability_distribution(y_prob, y_true, args.threshold, probability_distribution_path)

    print("\n=== Inference metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nRun output directory: {run_output_dir}")
    print(f"\nSaved per-image results: {per_image_path}")
    print(f"Saved detailed results: {detailed_results_path}")
    print(f"Saved legacy metrics: {legacy_metrics_path}")
    print(f"Saved args: {args_path}")
    print(f"Saved confusion matrix: {confusion_matrix_path}")
    print(f"Saved ROC curve: {roc_curve_path}")
    print(f"Saved probability distribution: {probability_distribution_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for ConvNeXt/ViT binary classifier (healthy vs anomalous)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="vindr",
        choices=["vindr", "inbreast"],
        help="Dataset type",
    )
    parser.add_argument(
        "--model-type",
        "--model_type",
        dest="model_type",
        type=str,
        required=True,
        choices=["convnext", "vit"],
        help="Classifier architecture",
    )
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        dest="checkpoint_path",
        type=str,
        required=True,
        help="Path to trained classifier checkpoint",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to evaluate",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=str,
        default=None,
        help="Root directory containing images",
    )
    parser.add_argument(
        "--metadata-path",
        "--metadata_path",
        dest="metadata_path",
        type=str,
        default=None,
        help="Path to metadata CSV",
    )

    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")

    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Directory to save outputs",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    dataset, data_dir, metadata_path = build_test_dataset(args)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Metadata path: {metadata_path}")
    print(f"Loaded test rows: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(args, device)
    y_true, y_prob, y_pred, image_names = run_inference(model, loader, device, args.threshold)

    metrics, y_pred = compute_metrics(y_true, y_prob, args.threshold)
    results_df = build_results_df(image_names, y_true, y_prob, y_pred)

    save_outputs(args, metrics, results_df, y_true, y_prob)


if __name__ == "__main__":
    main()
