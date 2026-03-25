import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Classifiers.models.breast_clip_classifier import BreastClipClassifier


BREAST_CLIP_ARCHS = {
    "upmc_breast_clip_det_b5_period_n_ft",
    "upmc_vindr_breast_clip_det_b5_period_n_ft",
    "upmc_breast_clip_det_b5_period_n_lp",
    "upmc_vindr_breast_clip_det_b5_period_n_lp",
    "upmc_breast_clip_det_b2_period_n_ft",
    "upmc_vindr_breast_clip_det_b2_period_n_ft",
    "upmc_breast_clip_det_b2_period_n_lp",
    "upmc_vindr_breast_clip_det_b2_period_n_lp",
}


class MammoInferenceDataset(Dataset):
    def __init__(self, args, df):
        self.args = args
        self.df = df.reset_index(drop=True)
        self.base_dir = Path(args.data_dir) / args.img_dir

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, row):
        image_id = str(row["image_id"])

        if "patient_id" in row.index and pd.notna(row["patient_id"]):
            patient_id = str(row["patient_id"])
            path = self.base_dir / patient_id / image_id
        else:
            path = self.base_dir / image_id

        if self.args.dataset.lower() == "rsna" and not str(path).endswith(".png"):
            path = Path(f"{path}.png")

        # For VinDr/custom CSVs where image_id may not include extension.
        if not path.exists() and not str(path).endswith(".png"):
            path_png = Path(f"{path}.png")
            if path_png.exists():
                path = path_png

        return path

    def _load_image(self, path):
        if self.args.arch.lower() in BREAST_CLIP_ARCHS:
            image = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
        else:
            image = np.array(Image.open(path).convert("L"), dtype=np.float32)

        image -= image.min()
        maxv = image.max()
        if maxv > 0:
            image /= maxv

        image = torch.tensor((image - self.args.mean) / self.args.std, dtype=torch.float32)
        return image.unsqueeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._resolve_path(row)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        x = self._load_image(path)
        y = float(row[self.args.label])

        return {
            "x": x,
            "y": y,
            "image_id": str(row["image_id"]),
            "img_path": str(path),
            "patient_id": str(row["patient_id"]) if "patient_id" in row.index else "",
            "laterality": str(row["laterality"]) if "laterality" in row.index else "",
        }


def collate_inference(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.float32),
        "image_id": [b["image_id"] for b in batch],
        "img_path": [b["img_path"] for b in batch],
        "patient_id": [b["patient_id"] for b in batch],
        "laterality": [b["laterality"] for b in batch],
    }


def prepare_inputs(x, arch):
    arch_l = arch.lower()
    if arch_l in BREAST_CLIP_ARCHS:
        return x.squeeze(1).permute(0, 3, 1, 2)
    if arch_l in {"swin_tiny_custom_norm", "swin_base_custom_norm", "swin_tiny_custom", "swin_base_custom"}:
        return x.squeeze(1)
    return x


def load_model(args, device):
    clip_ckpt = torch.load(args.clip_checkpoint_path, map_location="cpu", weights_only=False)
    model = BreastClipClassifier(args, ckpt=clip_ckpt, n_class=1)

    clf_ckpt = torch.load(args.classifier_checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = clf_ckpt["model"] if isinstance(clf_ckpt, dict) and "model" in clf_ckpt else clf_ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def build_test_df(args):
    csv_path = Path(args.data_dir) / args.csv_file
    df = pd.read_csv(csv_path).fillna(0)

    if args.label not in df.columns:
        if args.label == "anomaly" and "breast_birads" in df.columns:
            df["anomaly"] = (df["breast_birads"] != "BI-RADS 1").astype(int)
        else:
            raise ValueError(
                f"Label column '{args.label}' not found in CSV and no fallback available."
            )

    if "split" in df.columns:
        df = df[df["split"] == args.split].copy()

    if len(df) == 0:
        raise ValueError("No rows found for selected split/filters.")

    return df


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "n_samples": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    unique = np.unique(y_true)
    if len(unique) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None

    return metrics, y_pred


def run_inference(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    df_test = build_test_df(args)
    print(f"Loaded test rows: {len(df_test)}")

    dataset = MammoInferenceDataset(args, df_test)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_inference,
    )

    model = load_model(args, device)

    all_probs = []
    all_targets = []
    all_image_ids = []
    all_paths = []
    all_patient_ids = []
    all_laterality = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy().astype(int)

            x = prepare_inputs(x, args.arch)
            logits = model(x)
            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_targets.extend(y.tolist())
            all_image_ids.extend(batch["image_id"])
            all_paths.extend(batch["img_path"])
            all_patient_ids.extend(batch["patient_id"])
            all_laterality.extend(batch["laterality"])

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=np.float32)

    metrics, y_pred = compute_metrics(y_true, y_prob, args.threshold)

    results_df = pd.DataFrame(
        {
            "image_id": all_image_ids,
            "img_path": all_paths,
            "patient_id": all_patient_ids,
            "laterality": all_laterality,
            "true_label": y_true,
            "pred_prob_anomalous": y_prob,
            "pred_label": y_pred,
            "pred_class": np.where(y_pred == 1, "anomalous", "healthy"),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image_path = output_dir / "inference_per_image.csv"
    metrics_path = output_dir / "inference_metrics.json"

    results_df.to_csv(per_image_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Inference metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved per-image results: {per_image_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Mammo-CLIP downstream binary classifier (healthy vs anomalous)")

    parser.add_argument("--data-dir", type=str, required=True, help="Root directory containing csv and image folder")
    parser.add_argument("--img-dir", type=str, required=True, help="Image folder path relative to --data-dir")
    parser.add_argument("--csv-file", type=str, required=True, help="Metadata CSV path relative to --data-dir")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate (if CSV has split column)")
    parser.add_argument("--dataset", type=str, default="vindr", choices=["vindr", "rsna"], help="Dataset type for path handling")

    parser.add_argument("--clip-checkpoint-path", type=str, required=True, help="Path to Mammo-CLIP checkpoint used to initialize encoder")
    parser.add_argument("--classifier-checkpoint-path", type=str, required=True, help="Path to trained classifier checkpoint (.pth)")

    parser.add_argument("--arch", type=str, default="upmc_vindr_breast_clip_det_b5_period_n_ft", help="Architecture name used in training")
    parser.add_argument("--label", type=str, default="anomaly", help="Binary target column (0 healthy, 1 anomalous)")

    parser.add_argument("--mean", type=float, default=0.3089279, help="Normalization mean")
    parser.add_argument("--std", type=float, default=0.25053555408335154, help="Normalization std")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")

    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")

    args = parser.parse_args()
    run_inference(args)
