import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resize_preserve_aspect(image, target_size=(912, 1520), fill=(0, 0, 0)):
    """
    Resize image to fit within target_size while preserving aspect ratio,
    then pad to exactly target_size.

    target_size is (width, height).
    """
    target_w, target_h = target_size
    src_w, src_h = image.size

    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), fill)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


def _apply_transform(transform, image):
    """Apply either a torchvision-style or Albumentations-style transform."""
    if transform is None:
        return _resize_preserve_aspect(image.convert("RGB"))

    rgb_image = image.convert("RGB")
    rgb_image = _resize_preserve_aspect(rgb_image, target_size=(912, 1520))

    try:
        # Albumentations-style transform
        result = transform(image=np.asarray(rgb_image))
        image_out = result["image"] if isinstance(result, dict) else result

        if isinstance(image_out, np.ndarray):
            if image_out.ndim == 2:
                image_out = image_out[:, :, None]
            return torch.from_numpy(image_out).permute(2, 0, 1).float().div(255.0)

        return image_out

    except TypeError:
        # torchvision-style transform
        return transform(rgb_image)


class Inbreast_dataset(Dataset):
    def __init__(self, data_dir, metadata_path, transform=None, split="test"):
        """
        Dataset for classifier testing on INbreast.

        Args:
            data_dir: Path to INbreast images root.
            metadata_path: Path to inbreast_test_metadata.csv.
            transform: Optional torchvision transforms.
            split: Split to load. Defaults to "test".
        """
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.transform = transform
        self.split = split

        self.image_paths = []
        self.labels = []
        self.image_names = []

        self.df = pd.read_csv(metadata_path)
        self._load_data()

    def _get_label(self, row):
        """
        Return binary label (0 healthy, 1 anomalous).

        Priority:
        1) Use "anomaly" column if available.
        2) Fall back to breast_birads where BI-RADS 1 is healthy.
        """
        if "anomaly" in row.index:
            return int(row["anomaly"])

        birads = str(row["breast_birads"]).strip()
        return 0 if birads == "BI-RADS 1" else 1

    def _load_data(self):
        """Load INbreast rows for the requested split (test by default)."""
        df = self.df

        if "split" in df.columns:
            df = df[df["split"] == self.split]

        paths = []
        labels = []
        names = []

        for _, row in df.iterrows():
            image_id = row["image_id"]
            img_path = os.path.join(self.data_dir, image_id)

            if os.path.exists(img_path):
                paths.append(img_path)
                labels.append(self._get_label(row))
                names.append(image_id)
            else:
                print(f"Warning: INbreast image not found: {img_path}")

        self.image_paths = paths
        self.labels = labels
        self.image_names = names

        print(f"Loaded {len(self.image_paths)} INbreast images (split: {self.split})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image_name = self.image_names[idx]

        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")

        image = _apply_transform(self.transform, image)

        return image, torch.tensor(label, dtype=torch.float32), image_name

    def get_class_distribution(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def get_split_info(self):
        return {
            "split": self.split,
            "total_samples": len(self.image_paths),
            "class_distribution": self.get_class_distribution(),
        }
