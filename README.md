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

The scripts import local paths from `src/config.py`. This file is intentionally
ignored because dataset and model locations are machine-specific. Create it
from the tracked example, then change the dataset-specific variables:

```bash
cp src/config.example.py src/config.py
```

You can also set `PROJECT_ROOT` to place the `data/` and `models/` directories
outside the repository. Do not commit the copied `src/config.py`.

## Hugging Face Artifacts

The trained diffusion checkpoint is available in the [DDPM healthy mammogram
generation repository](https://huggingface.co/senigarcia/ddpm-healthy-mammogram-generation):

- `model008000.pt`: trained DDPM checkpoint for healthy mammogram generation.
- `training_args.json`: training arguments associated with the checkpoint.

The classifier checkpoints and their exact training arguments are available in
the [healthy-vs-anomalous classifier checkpoint repository](https://huggingface.co/senigarcia/healthy_vs_anomalous-classifier-checkpoints):

- `checkpoints/convnext_baseline.pth` and `configs/convnext_baseline_args.json`:
  ConvNeXt trained without counterfactual augmentation.
- `checkpoints/convnext_cf.pth` and `configs/convnext_cf_args.json`:
  ConvNeXt trained with counterfactual augmentation.
- `checkpoints/vit_baseline.pth` and `configs/vit_baseline_args.json`:
  ViT trained without counterfactual augmentation.
- `checkpoints/vit_cf.pth` and `configs/vit_cf_args.json`:
  ViT trained with counterfactual augmentation.
- `checkpoints/mammo_clip_baseline.pth` and `configs/mammo_clip_baseline_args.json`:
  Mammo-CLIP trained without counterfactual augmentation.
- `checkpoints/mammo_clip_cf.pth` and `configs/mammo_clip_cf_args.json`:
  Mammo-CLIP trained with counterfactual augmentation.
- `checkpoints/fpn_mil_baseline_closest.pth` and
  `configs/fpn_mil_baseline_closest_args.json`: closest surviving FPN-MIL
  baseline checkpoint to the reported table metrics.
- `checkpoints/fpn_mil_cf_closest.pth` and
  `configs/fpn_mil_cf_closest_args.json`: closest surviving FPN-MIL
  counterfactual-augmentation checkpoint to the reported table metrics.

The original FPN-MIL checkpoint used for the reported table was stored on
another server that was eventually erased. The uploaded FPN-MIL checkpoints
are explicitly labeled as the closest surviving runs, not as the confirmed
source of the table results. See `configs/fpn_mil_checkpoint_note.md` in the
Hugging Face repository for the provenance note.

The image datasets are available in the [Hugging Face mammogram dataset
repository](https://huggingface.co/datasets/senigarcia/diffusion-based-counterfactual-generation).
They are stored as separate directories:

- `VinDrMammo-CLIP-512-CLAHE`: processed VinDrMammo images used by the classifiers.
- `VinDrMammo-CLIP-512-CLAHE-912-1520`: resized images used for full-resolution evaluation and FID preparation.
- `repaint_results`: generated counterfactual images used for classifier augmentation and assessment.
- `repaint_results_912x1520`: resized counterfactual images used mainly for FID preparation.

The same folders can be uploaded or updated with
`src/E_Aux_Scripts/upload_image_datasets_to_hf.py`.

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

- **Preprocess images**: paths default to the local `src/config.py` values and
  can be overridden explicitly:
  ```bash
  python src/B_Dataset_Preprocessing/images_preprocessing.py \
    --operation process
  python src/B_Dataset_Preprocessing/images_preprocessing.py \
    --operation upscale-counterfactuals
  ```

- **Preprocess metadata**:
  ```bash
  python src/B_Dataset_Preprocessing/metadata_preprocessing.py \
    --label birads
  ```
  Use `--input-csv`, `--output-json`, `--image-dir`, and `--cf-dir` when
  working with a different dataset or directory layout.

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

This work uses the counterfactual-generation technique described in the
[prototype-assertive-proactive repository](https://github.com/MIMBCD-UI/prototype-assertive-proactive#citing).

This work was presented at the DeepBreast Workshop at MICCAI 2026 in
Strasbourg.

## License
TO DO
