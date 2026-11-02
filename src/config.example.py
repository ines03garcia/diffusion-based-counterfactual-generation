"""Example local configuration.

Copy this file to ``src/config.py`` and update the dataset-specific directory
names and paths for your environment. The copied file is ignored by Git.
"""

import os


# The default resolves to the repository containing this file. Set
# PROJECT_ROOT when data and models live outside the repository.
ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(ROOT, "data")
CODE_ROOT = os.path.join(ROOT, "src")
MODELS_ROOT = os.path.join(ROOT, "models")

IMAGES_ROOT = os.path.join(DATA_ROOT, "images")
METADATA_ROOT = os.path.join(DATA_ROOT, "metadata")
LOGS_PATH = os.path.join(DATA_ROOT, "logs")

# Change these dataset-specific names to match your local files.
DATASET_DIR = os.path.join(IMAGES_ROOT, "dataset")
RESIZED_DATASET_DIR = os.path.join(IMAGES_ROOT, "dataset_resized")
PROCESSED_DATASET_DIR = os.path.join(IMAGES_ROOT, "dataset_processed")
CF_DIR = os.path.join(IMAGES_ROOT, "counterfactuals")
MASKS_DIR = os.path.join(IMAGES_ROOT, "masks")