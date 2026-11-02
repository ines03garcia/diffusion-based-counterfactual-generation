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
CLASSIFIER_CHECKPOINTS_DIR = MODELS_ROOT

# Change these dataset-specific names to match your local files.
DATASET_DIR = os.path.join(IMAGES_ROOT, "dataset")
RESIZED_DATASET_DIR = os.path.join(IMAGES_ROOT, "dataset_resized")
PROCESSED_DATASET_DIR = os.path.join(IMAGES_ROOT, "dataset_processed")
UPSCALED_DATASET_DIR = os.path.join(IMAGES_ROOT, "dataset_processed_upscaled")
INBREAST_DATASET_DIR = os.path.join(IMAGES_ROOT, "datasets", "inbreast")
CF_DIR = os.path.join(IMAGES_ROOT, "counterfactuals")
UPSCALED_CF_DIR = os.path.join(IMAGES_ROOT, "counterfactuals_upscaled")
MASKS_DIR = os.path.join(IMAGES_ROOT, "masks")

GROUPED_METADATA_PATH = os.path.join(METADATA_ROOT, "grouped_df.csv")
PROCESSED_METADATA_PATH = os.path.join(METADATA_ROOT, "processed_df_birads.json")
PROCESSED_METADATA_512_PATH = os.path.join(METADATA_ROOT, "processed_df_birads_512.json")
RESIZED_METADATA_PATH = os.path.join(METADATA_ROOT, "resized_df_512.json")
INBREAST_METADATA_PATH = os.path.join(METADATA_ROOT, "inbreast_test_metadata.csv")

MAMMO_CLIP_B2_CHECKPOINT = os.path.join(CLASSIFIER_CHECKPOINTS_DIR, "b2-model-best-epoch-10.tar")
MAMMO_CLIP_B5_CHECKPOINT = os.path.join(CLASSIFIER_CHECKPOINTS_DIR, "b5-model-best-epoch-7.tar")

FID_PATCHES_DIR = os.path.join(IMAGES_ROOT, "patches_for_fid")
FID_CF_PATCHES_DIR = os.path.join(FID_PATCHES_DIR, "cf_patches", "repaint_patches_512")
FID_HEALTHY_PATCHES_DIR = os.path.join(
	FID_PATCHES_DIR, "healthy_patches", "healthy_patches_for_fid_512"
)
HEALTHY_FID_DIR = os.path.join(IMAGES_ROOT, "healthy_for_fid")
TEST_CF_FID_DIR = os.path.join(IMAGES_ROOT, "test_cf_for_fid")
TEST_HEALTHY_FID_DIR = os.path.join(IMAGES_ROOT, "test_healthy_for_fid")
ATTENTION_IOU_LOGS_DIR = os.path.join(LOGS_PATH, "attention_iou")

RADIOLOGIST_OUTPUT_DIR = os.path.join(IMAGES_ROOT, "radiologist_assessment", "full_realism")
RADIOLOGIST_BBOX_OUTPUT_DIR = os.path.join(
	IMAGES_ROOT, "radiologist_assessment", "healthy_patch_realism"
)
RADIOLOGIST_METADATA_PATH = os.path.join(METADATA_ROOT, "radiologist_dataset.json")
NEW_RADIOLOGIST_TASK1_PATH = os.path.join(METADATA_ROOT, "new_radiologist_assessment_task1.csv")
NEW_RADIOLOGIST_TASK2_PATH = os.path.join(METADATA_ROOT, "new_radiologist_assessment_task2.csv")
NEW_RADIOLOGIST_RESULTS_DIR = os.path.join(DATA_ROOT, "results", "new_radiologist_assessment")