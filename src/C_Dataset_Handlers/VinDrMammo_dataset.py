import os
import json
import logging
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

class VinDrMammo_dataset(Dataset):
    def __init__(self, split=None, label=None, cf_dir=None, cv_fold=0, data_dir=None, metadata_path=None, transform=None):
        """
        Args:
        - split: 'train', 'test', 'val' or None
        - label: 'all', 'healthy', 'anomalous', 'anomalous_with_findings' or None
        - cf_dir: counterfactuals directory or None if not using counterfactuals
        - cv_fold: validation fold (0, 1, 2, 3), the remaining are for training
        - data_dir: dataset directory
        - metadata_path: path to metadata JSON file
        - transform: torchvision transforms to apply to the images
        """
        self.split = split
        self.label = label
        self.cf_dir = cf_dir
        self.cv_fold = cv_fold
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.transform = transform

        # Initialize data lists
        self.image_paths = []
        self.labels = []
        self.image_ids = []
        
        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.data = json.load(f)

        # Load data
        self._load_data()

        logger.info(f"Loaded {len(self.image_paths)} samples (split: {self.split}, label: {self.label})")
        logger.info(f"Class distribution: {sum(1 for l in self.labels if l==0)} healthy, {sum(1 for l in self.labels if l==1)} anomalous")

    def _load_data(self):
        # Filter data based on split
        if self.split == "train":
            records_filtered_by_split = [r for r in self.data if r.get("split") == "train" and r.get("fold") != self.cv_fold]
        elif self.split == "val":
            records_filtered_by_split = [r for r in self.data if r.get("split") == "train" and r.get("fold") == self.cv_fold]
        elif self.split == "test":
            records_filtered_by_split = [r for r in self.data if r.get("split") == "test"]
        else: # None
            records_filtered_by_split = self.data

        # Filer data based on label
        if self.label == "all":
            records_filtered_by_label = records_filtered_by_split
        elif self.label == "healthy":
            records_filtered_by_label = [r for r in records_filtered_by_split if r.get("healthy") == 1]
        elif self.label == "anomalous":
            records_filtered_by_label = [r for r in records_filtered_by_split if r.get("healthy") == 0]
        elif self.label == "anomalous_with_findings":
            records_filtered_by_label = [r for r in records_filtered_by_split if r.get("has_cf") == 1]
        else: # None
            records_filtered_by_label = []
            logger.warning("No label label specified, original images will not be included.")
            
        # Add images paths, labels, and names
        for item in records_filtered_by_label:
            image_path = os.path.join(self.data_dir, item['image_id'])
            self.image_paths.append(image_path)
            self.labels.append(1 - item['healthy'])
            self.image_ids.append(item['image_id'])

        if self.cf_dir:
            logger.info("Adding counterfactuals...")
            records_filtered_by_cf = [r for r in records_filtered_by_split if r.get("has_cf") == 1]
            for item in records_filtered_by_cf:
                cf_image_path = os.path.join(self.cf_dir, item['image_id'])
                self.image_paths.append(cf_image_path)
                self.labels.append(0)
                self.image_ids.append(item['image_id']) # Same id as normal images, but different folder
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image_id = self.image_ids[idx]

        # Load image as grayscale (original)
        image = Image.open(image_path).convert("L")

        # Apply transforms (e.g. convert to RGB, resize, ...)
        if self.transform:
            image = self.transform(image)

        return image, label, image_id
    
    def __len__(self):
        return len(self.image_paths)
        
        