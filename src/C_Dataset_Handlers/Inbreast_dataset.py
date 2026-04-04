import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

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
