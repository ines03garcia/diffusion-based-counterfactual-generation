# Healthy Counterfactual Generation via Diffusion Inpainting for Mammography Classification

This repository provides code for generating counterfactual images using diffusion models and its application through data augmentation in deep learning mammogram classifiers. The project is organized for research in medical imaging, specifically mammography, but can be adapted for other domains.

## Project Structure

- **src/**
  - **A_Dataset_Exploration/**
    - Jupyter notebook to explore the original metadata.
  - **B_Dataset_Preprocessing/**
    - Scripts to preprocess the images and the metadata.
  - **C_Dataset_Handlers/**
    - Dataset classes (VinDrMammo and INBreast)
  - **D_Models/**
    - Model specific classes and logic (ConvNeXt, ViT, Mammo-CLIP and FPN-MIL)
  - **E_Aux_Scripts/**
    - utility and logger functions
  - **F_Tasks/**
    - task scripts (train classifier, test classifier, ...)

- **data/**
  - `images/`: Contains dataset images, generated counterfactuals, masks, explainability visualizations, besides others. In particular, the images under `images/datasets/images_png/` are the original Mammo-CLIP images, sourced from https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png.
  - `logs/`: logs for each task experiment that was run.
  - `metadata/`: folder with original metadata, json files with processed metadata, radiologist assessment information, besides others. In particular, `grouped_df.csv` comes from the repository https://github.com/marianamourao-37/Multi-scale-Attention-based-MIL and is the grouped-by-`image_id` version of the Mammo-CLIP metadata file `src/codebase/data_csv/vindr_detection_v1_folds.csv`.
  - `zips/`: Compressed data files.

- **models/**
  - DDPM checkpoint.
  - Pretrained model weights for classifiers.

## Configuration

A `config.py` file should define the project paths used by the scripts. For example:

```python
import os

ROOT = "path/to/diffusion-based-counterfactual-generation"
DATA_ROOT = os.path.join(ROOT, "data")
CODE_ROOT = os.path.join(ROOT, "code")
MODELS_ROOT = os.path.join(ROOT, "models")

IMAGES_ROOT = os.path.join(DATA_ROOT, "images")
METADATA_ROOT = os.path.join(DATA_ROOT, "metadata")
LOGS_PATH = os.path.join(DATA_ROOT, "logs")

RESIZED_DATASET_DIR = os.path.join(IMAGES_ROOT, "VinDr-Mammo-Clip-512")
PROCESSED_DATASET_DIR = os.path.join(IMAGES_ROOT, "VinDr-Mammo-Clip-CLAHE-512")
CF_DIR = os.path.join(IMAGES_ROOT, "repaint_results")
MASKS_DIR = os.path.join(IMAGES_ROOT, "masks_512")
```

## Key Features

- **Diffusion Models**: Implementation of DDPM and guided diffusion for counterfactual image generation.
- **Classification**: Deep learning classifiers (ConvNeXt, ViT, Mammo-CLIP and FPN-MIL) for integrating the generated images as data augmentation.
- **Evaluation**: explainability and radiologist assessment interpretation scripts.
- **Medical Imaging Focus**: Tools and scripts tailored for the VinDr-Mammo mammography dataset, but can easily be adapted.

## Setup

1. **Clone the repository**
2. **Install dependencies** (recommended: use a virtual environment)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Prepare data**: Place images and metadata in the `data/` folder as described above.
4. **Download models**: Place pretrained weights in the `models/` folder.

## Usage

- **Train diffusion models**: `src/DDPM/scripts/image_train.py`
  ```bash
  python src/DDPM/scripts/image_train.py [training-parameters]
  ```

- **Sample images with a trained diffusion model**: `src/DDPM/scripts/image_sample.py`
  ```bash
  python src/DDPM/scripts/image_sample.py [sampling-parameters]
  ```

- **Train deep learning classifiers**: `src/F_Tasks/train_classifier.py`
  ```bash
  python src/F_Tasks/train_classifier.py [mode] [augmentation] <model_type> [model-specific-parameters]
  ```
  Training supports three mutually exclusive execution modes:
  - `--cross-validation`
  - `--multiple_seeds`
  - `--single_seed`

  Optional training flags include:
  - `--use_counterfactuals` to add counterfactual samples to the training set.
  - `--add_cf_batch` to pair originals with their counterfactuals inside batches.
  - `--no_validation` for training without a validation split or early stopping.
  - `--loss low` to train with the LOW-based loss instead of BCE.
  - `--resume_from_checkpoint` to continue from an existing checkpoint.

  Examples:
  ```bash
  python src/F_Tasks/train_classifier.py --single_seed convnext --epochs 27
  python src/F_Tasks/train_classifier.py --cross-validation --use_counterfactuals vit --epochs 9
  python src/F_Tasks/train_classifier.py --multiple_seeds --add_cf_batch mammo-clip --epochs 24
  ```

- **Test deep learning classifiers**: `src/F_Tasks/test_classifier.py`
  ```bash
  python src/F_Tasks/test_classifier.py <model_type> [test-parameters]
  ```
  The test script now supports:
  - `--cross-validation` to evaluate fold checkpoints from a checkpoint directory.
  - `--multiple_seeds` to evaluate all `seed_*/final_model.pth` checkpoints under a checkpoint root.
  - `--aggregate_results` to save mean/std summaries across folds or seeds.
  - `--fixed_specificity` and `--fixed_specificity_value` to evaluate at a chosen operating point instead of the full metric suite.
  - `--output_path` to aggregate previously generated results without rerunning inference.
  - `--use_counterfactuals` to include counterfactual samples during test loading for VinDr.

  Examples:
  ```bash
  python src/F_Tasks/test_classifier.py convnext --checkpoint_path models/baseline_aug/seed_0/final_model.pth
  python src/F_Tasks/test_classifier.py --cross-validation vit --checkpoint_dir models/baseline_aug
  python src/F_Tasks/test_classifier.py --multiple_seeds mammo-clip --checkpoint_path data/logs/Classifiers/train/mammo-clip/baseline_aug/multiple_seeds/local_00/
  ```

- **Compare baseline vs counterfactual augmentation results**: `src/F_Tasks/test_classifier_significance.py`
  ```bash
  python src/F_Tasks/test_classifier_significance.py --baseline_aug <baseline_results_dir> --cf_aug <cf_results_dir> [--metrics ...] [--output_path ...]
  ```
  This script compares matching `test_metrics.json` files with ASO and can save a JSON summary for downstream analysis.

## Citation
TO DO
## License
TO DO
