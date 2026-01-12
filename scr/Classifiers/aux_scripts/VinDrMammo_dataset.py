import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from PIL import Image

class VinDrMammo_dataset(Dataset):
    def __init__(self, data_dir, metadata_path, split="training", transform=None, 
                 use_counterfactuals=False, counterfactuals_dir=None,
                 training_category=None, training_cf=False, testing_category=None, testing_cf=False,
                 anomaly_type="birads"):
        """
        Args:
            data_dir: path to the dataset root directory (DATA_ROOT)
            metadata_path: path to the counterfactuals_df.csv file
            split: "training", "test", "validation", or "all" - determines which data splits to load
            transform: data transformations to apply
            use_counterfactuals: whether to include counterfactual images (backward compatibility)
            counterfactuals_dir: directory containing counterfactual images
            training_category: filter training data by category ("healthy", "anomalous", "anomalous_with_findings", or None for all)
            training_cf: whether to include counterfactuals in training data
            testing_category: filter test data by category ("healthy", "anomalous", "anomalous_with_findings", or None for all)
            testing_cf: whether to include counterfactuals in test data
            anomaly_type: type of anomaly classification ("birads", "mass", or "calcification")
        """
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.split = split
        self.transform = transform
        self.use_counterfactuals = use_counterfactuals
        self.counterfactuals_dir = counterfactuals_dir
        
        self.training_category = training_category  # "healthy", "anomalous", "anomalous_with_findings", or None
        self.training_cf = training_cf # Include training counterfactuals
        self.testing_category = testing_category  # "healthy", "anomalous", "anomalous_with_findings", or None
        self.testing_cf = testing_cf # Include testing counterfactuals
        
        # Validate and store anomaly type
        if anomaly_type not in ["birads", "mass", "calcification"]:
            raise ValueError("Invalid anomaly_type. Choose from 'birads', 'mass', or 'calcification'.")
        self.anomaly_type = anomaly_type
        
        # Initialize data lists
        self.image_paths = []
        self.labels = []
        self.image_names = []
        
        # Load metadata
        self.df = pd.read_csv(metadata_path)
        
        # Load data and create train/val split if needed
        self._load_data()
    
    def _is_healthy(self, row):
        """
        Determine if a data row represents a healthy case based on the anomaly_type.
        
        Args:
            row: pandas Series representing a single row from the metadata dataframe
            
        Returns:
            bool: True if healthy, False if anomalous
        """
        if self.anomaly_type == "birads":
            return row['breast_birads'] == 'BI-RADS 1'
        elif self.anomaly_type == "mass":
            return row['Mass'] == 0
        elif self.anomaly_type == "calcification":
            return row['Suspicious_Calcification'] == 0
        else:
            raise ValueError(f"Invalid anomaly_type: {self.anomaly_type}")
    
    def _get_label(self, row):
        """
        Get the binary label (0 for healthy, 1 for anomalous) for a data row.
        
        Args:
            row: pandas Series representing a single row from the metadata dataframe
            
        Returns:
            int: 0 for healthy, 1 for anomalous
        """
        return 0 if self._is_healthy(row) else 1
    
    def load_train_data(self):
        """Load training data with optional category filtering and counterfactuals"""
        train_df = self.df[self.df['split'] == 'training']
        
        # Apply category filter if specified
        if self.training_category == "healthy":
            train_df = train_df[train_df.apply(self._is_healthy, axis=1)]
        elif self.training_category == "anomalous":
            train_df = train_df[~train_df.apply(self._is_healthy, axis=1)]
        elif self.training_category == "anomalous_with_findings":
            # Anomalous cases that have counterfactuals (meaning they have findings)
            train_df = train_df[(~train_df.apply(self._is_healthy, axis=1)) & (train_df['has_counterfactual'] == 1)]
        # If None, include all categories
        
        train_paths = []
        train_labels = []
        train_names = []
        
        # Load original training images
        for _, row in train_df.iterrows():
            img_path = os.path.join(self.data_dir, row['image_id'])
            if os.path.exists(img_path):
                train_paths.append(img_path)
                label = self._get_label(row)
                train_labels.append(label)
                train_names.append(row['image_id'])
            else:
                print(f"Warning: Training image not found: {img_path}")
        
        # Add counterfactuals if requested
        if self.training_cf and self.counterfactuals_dir:
            cf_df = train_df[train_df['has_counterfactual'] == 1]
            for _, row in cf_df.iterrows():
                cf_path = os.path.join(self.counterfactuals_dir, row['image_id'])
                if os.path.exists(cf_path):
                    train_paths.append(cf_path)
                    train_labels.append(0)  # Counterfactuals are healthy
                    train_names.append(row['image_id'])
                else:
                    print(f"Warning: Counterfactual image not found: {cf_path}")
        
        print(f"Loaded {len(train_paths)} training images (category: {self.training_category or 'all'}, counterfactuals: {self.training_cf})")
        return train_paths, train_labels, train_names
    
    def load_test_data(self):
        """Load test data with optional category filtering and counterfactuals"""
        test_df = self.df[self.df['split'] == 'test']
        
        # Apply category filter if specified
        if self.testing_category == "healthy":
            test_df = test_df[test_df.apply(self._is_healthy, axis=1)]
        elif self.testing_category == "anomalous":
            test_df = test_df[~test_df.apply(self._is_healthy, axis=1)]
        elif self.testing_category == "anomalous_with_findings":
            # Anomalous cases that have counterfactuals (meaning they have findings)
            test_df = test_df[(~test_df.apply(self._is_healthy, axis=1)) & (test_df['has_counterfactual'] == 1)]
        # If None, include all categories
        
        test_paths = []
        test_labels = []
        test_names = []
        
        if self.testing_category != "counterfactuals_only":
            # Load original test images
            for _, row in test_df.iterrows():
                img_path = os.path.join(self.data_dir, row['image_id'])
                if os.path.exists(img_path):
                    test_paths.append(img_path)
                    label = self._get_label(row)
                    test_labels.append(label)
                    test_names.append(row['image_id'])
                else:
                    print(f"Warning: Test image not found: {img_path}")
        
        # Add counterfactuals if requested
        if self.testing_cf and self.counterfactuals_dir:
            cf_df = test_df[test_df['has_counterfactual'] == 1]
            for _, row in cf_df.iterrows():
                cf_path = os.path.join(self.counterfactuals_dir, row['image_id'])
                if os.path.exists(cf_path):
                    test_paths.append(cf_path)
                    test_labels.append(0)  # Counterfactuals are always healthy
                    test_names.append(row['image_id'])
                else:
                    print(f"Warning: Counterfactual image not found: {cf_path}")
        
        print(f"Loaded {len(test_paths)} test images (category: {self.testing_category or 'all'}, counterfactuals: {self.testing_cf})")
        return test_paths, test_labels, test_names
    
    def load_validation_data(self):
        """Load validation data with optional category filtering and counterfactuals"""
        val_df = self.df[self.df['split'] == 'validation']
        
        # Apply category filter if specified (use testing category for validation)
        if self.testing_category == "healthy":
            val_df = val_df[val_df.apply(self._is_healthy, axis=1)]
        elif self.testing_category == "anomalous":
            val_df = val_df[~val_df.apply(self._is_healthy, axis=1)]
        elif self.testing_category == "anomalous_with_findings":
            # Anomalous cases that have counterfactuals (meaning they have findings)
            val_df = val_df[(~val_df.apply(self._is_healthy, axis=1)) & (val_df['has_counterfactual'] == 1)]
        # If None, include all categories
        
        val_paths = []
        val_labels = []
        val_names = []
        
        # Load original validation images
        for _, row in val_df.iterrows():
            img_path = os.path.join(self.data_dir, row['image_id'])
            if os.path.exists(img_path):
                val_paths.append(img_path)
                label = self._get_label(row)
                val_labels.append(label)
                val_names.append(row['image_id'])
            else:
                print(f"Warning: Validation image not found: {img_path}")
        
        # Add counterfactuals if requested (use testing_cf flag for validation)
        if self.testing_cf and self.counterfactuals_dir:
            cf_df = val_df[val_df['has_counterfactual'] == 1]
            for _, row in cf_df.iterrows():
                cf_path = os.path.join(self.counterfactuals_dir, row['image_id'])
                if os.path.exists(cf_path):
                    val_paths.append(cf_path)
                    val_labels.append(0)  # Counterfactuals are always healthy
                    val_names.append(row['image_id'])
                else:
                    print(f"Warning: Counterfactual image not found: {cf_path}")
        
        print(f"Loaded {len(val_paths)} validation images (category: {self.testing_category or 'all'}, counterfactuals: {self.testing_cf})")
        return val_paths, val_labels, val_names
    
    def load_all_data(self):
        """Load training, validation, and test data according to their respective flags"""
        all_paths = []
        all_labels = []
        all_names = []
        
        # Load training data
        train_paths, train_labels, train_names = self.load_train_data()
        all_paths.extend(train_paths)
        all_labels.extend(train_labels)
        all_names.extend(train_names)
        
        # Load validation data
        val_paths, val_labels, val_names = self.load_validation_data()
        all_paths.extend(val_paths)
        all_labels.extend(val_labels)
        all_names.extend(val_names)
        
        # Load test data
        test_paths, test_labels, test_names = self.load_test_data()
        all_paths.extend(test_paths)
        all_labels.extend(test_labels)
        all_names.extend(test_names)
        
        print(f"Loaded {len(all_paths)} total images (train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)})")
        return all_paths, all_labels, all_names
        
    def _load_data(self):
        """Load the dataset based on split type and flags"""
        if self.split == "training":
            # Load only training data
            paths, labels, names = self.load_train_data()
            self.image_paths = paths
            self.labels = labels
            self.image_names = names
        elif self.split == "test":
            # Load only test data
            paths, labels, names = self.load_test_data()
            self.image_paths = paths
            self.labels = labels
            self.image_names = names
        elif self.split == "validation":
            # Load only validation data
            paths, labels, names = self.load_validation_data()
            self.image_paths = paths
            self.labels = labels
            self.image_names = names
        elif self.split == "all":
            # Load both training and test data
            paths, labels, names = self.load_all_data()
            self.image_paths = paths
            self.labels = labels
            self.image_names = names
        else:
            raise ValueError(f"Invalid split: {self.split}. Choose from 'training', 'test', 'validation', or 'all'.")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image_name = self.image_names[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32), image_name
    
    def get_class_distribution(self):
        """Get the distribution of classes in the current split"""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        return distribution
    
    def get_split_info(self):
        """Get information about the current split"""
        total_samples = len(self.image_paths)
        class_dist = self.get_class_distribution()
        
        info = {
            'split': self.split,
            'total_samples': total_samples,
            'class_distribution': class_dist,
            'uses_counterfactuals': self.use_counterfactuals,
            'anomaly_type': self.anomaly_type,
            'training_category': self.training_category,
            'training_cf': self.training_cf,
            'testing_category': self.testing_category,
            'testing_cf': self.testing_cf
        }
        
        return info
    
    def get_config_summary(self):
        """Get a summary of the current dataset configuration"""
        config = {
            'split': self.split,
            'total_samples': len(self.image_paths),
            'class_distribution': self.get_class_distribution(),
            'anomaly_type': self.anomaly_type,
            'training_config': {
                'category_filter': self.training_category or 'all',
                'include_counterfactuals': self.training_cf
            },
            'testing_config': {
                'category_filter': self.testing_category or 'all', 
                'include_counterfactuals': self.testing_cf
            }
        }
        return config
    
    def get_sample_info(self, idx):
        """Get detailed information about a specific sample"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image_name = self.image_names[idx]
        
        # Check if this is a counterfactual image
        is_counterfactual = 'counterfactuals' in img_path
        
        return {
            'index': idx,
            'image_path': img_path,
            'label': label,
            'image_name': image_name,
            'is_counterfactual': is_counterfactual,
            'split': self.split
        }
    
    def load_test_anomalous_with_counterfactuals(self):
        """
        Load test split anomalous images that have counterfactuals available.
        Returns a list of dictionaries with original image info and counterfactual path.
        """
        # Filter for test split anomalous images with counterfactuals
        test_anomalous_df = self.df[
            (self.df['split'] == 'test') & 
            (~self.df.apply(self._is_healthy, axis=1)) & 
            (self.df['has_counterfactual'] == 1)
        ]
        
        anomalous_data = []
        
        for _, row in test_anomalous_df.iterrows():
            # Original image path
            original_img_path = os.path.join(self.data_dir, row['image_id'])
            
            # Counterfactual image path
            cf_img_path = os.path.join(self.counterfactuals_dir, row['image_id']) if self.counterfactuals_dir else None
            
            # Check if both files exist
            if os.path.exists(original_img_path) and (cf_img_path is None or os.path.exists(cf_img_path)):
                anomalous_data.append({
                    'image_id': row['image_id'],
                    'original_path': original_img_path,
                    'counterfactual_path': cf_img_path,
                    'breast_birads': row['breast_birads'],
                    'finding_categories': row['finding_categories'],
                    'patient_id': row['patient_id'],
                    'laterality': row['laterality'],
                    'view': row['view']
                })
            else:
                missing_files = []
                if not os.path.exists(original_img_path):
                    missing_files.append(f"original: {original_img_path}")
                if cf_img_path and not os.path.exists(cf_img_path):
                    missing_files.append(f"counterfactual: {cf_img_path}")
                print(f"Warning: Missing files for {row['image_id']}: {', '.join(missing_files)}")
        
        print(f"Loaded {len(anomalous_data)} test anomalous images with counterfactuals")
        return anomalous_data
    
    def get_image_and_counterfactual(self, image_id):
        """
        Load both the original image and its counterfactual for a given image_id.
        Returns tuple: (original_image, counterfactual_image, label)
        """
        # Find the image in the dataframe
        row = self.df[self.df['image_id'] == image_id]
        if row.empty:
            raise ValueError(f"Image {image_id} not found in metadata")
        
        row = row.iloc[0]
        
        # Load original image
        original_path = os.path.join(self.data_dir, image_id)
        original_image = Image.open(original_path).convert('RGB')
        
        # Load counterfactual image if available
        counterfactual_image = None
        if self.counterfactuals_dir and row['has_counterfactual'] == 1:
            cf_path = os.path.join(self.counterfactuals_dir, image_id)
            if os.path.exists(cf_path):
                counterfactual_image = Image.open(cf_path).convert('RGB')
            else:
                print(f"Warning: Counterfactual not found at {cf_path}")
        
        # Determine label (anomalous = 1, healthy = 0)
        label = self._get_label(row)
        
        # Apply transforms if available
        if self.transform:
            original_image = self.transform(original_image)
            if counterfactual_image is not None:
                counterfactual_image = self.transform(counterfactual_image)
        
        return original_image, counterfactual_image, label