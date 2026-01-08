# Example usage of the new VinDrMammo_dataset with flags

import sys
import os
sys.path.append('/projects/F202507605CPCAA0/inescgarcia/thesis/code/Classifiers')

from aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset
from aux_scripts.config import DATA_ROOT, DATA_DIR, METADATA_ROOT

# Example configurations:

print("=== Example 1: Training data only, all categories, with counterfactuals ===")
dataset1 = VinDrMammo_dataset(
    data_dir=DATA_ROOT,
    metadata_path=os.path.join(METADATA_ROOT, "resized_df_counterfactuals.csv"),
    split="training",
    training_category=None,  # all categories
    training_cf=True,       # include counterfactuals
    counterfactuals_dir=os.path.join(DATA_DIR, "counterfactuals")
)
print(f"Dataset 1 info: {dataset1.get_config_summary()}")

print("\n=== Example 2: Test data only, anomalous cases only, no counterfactuals ===")
dataset2 = VinDrMammo_dataset(
    data_dir=DATA_ROOT,
    metadata_path=os.path.join(METADATA_ROOT, "resized_df_counterfactuals.csv"),
    split="test",
    testing_category="anomalous",  # only anomalous cases
    testing_cf=False,              # no counterfactuals
    counterfactuals_dir=os.path.join(DATA_DIR, "counterfactuals")
)
print(f"Dataset 2 info: {dataset2.get_config_summary()}")

print("\n=== Example 3: All data, training=healthy+CF, testing=anomalous only ===")
dataset3 = VinDrMammo_dataset(
    data_dir=DATA_ROOT,
    metadata_path=os.path.join(METADATA_ROOT, "resized_df_counterfactuals.csv"),
    split="all",
    training_category="healthy",    # only healthy training cases
    training_cf=True,              # with counterfactuals
    testing_category="anomalous",  # only anomalous test cases
    testing_cf=False,              # no test counterfactuals
    counterfactuals_dir=os.path.join(DATA_DIR, "counterfactuals")
)
print(f"Dataset 3 info: {dataset3.get_config_summary()}")

print("\n=== Example 4: Test anomalous with findings only ===")
dataset4 = VinDrMammo_dataset(
    data_dir=DATA_ROOT,
    metadata_path=os.path.join(METADATA_ROOT, "resized_df_counterfactuals.csv"),
    split="test",
    testing_category="anomalous_with_findings",  # only anomalous cases that have findings/counterfactuals
    testing_cf=False,                           # no counterfactuals in dataset
    counterfactuals_dir=os.path.join(DATA_DIR, "counterfactuals")
)
print(f"Dataset 4 info: {dataset4.get_config_summary()}")