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
  - `images/`: Contains dataset images, generated counterfactuals, masks, explainability visualizations, besides others.
  - `logs/`: logs for each task experiment that was run.
  - `metadata/`: folder with original metadata (grouped_df.csv), json file with processed metadata, radiologist assessment information, besides others.
  - `zips/`: Compressed data files.

- **models/**
  - DDPM checkpoint.
  - Pretrained model weights for classifiers.

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

- **Train deep learning classifiers (with or without counterfactual augmentation)**: `src/F_Tasks/train_classifier.py`
  ```bash
  python src/F_Tasks/train_classifier.py [--cross-validation] [--use_counterfactuals] <model_type> [model-specific-parameters]
  ```
  Example:
  ```bash
  python src/F_Tasks/train_classifier.py convnext --use_counterfactuals
  ```

- **Test deep learning classifiers**: `src/F_Tasks/test_classifier.py`
  ```bash
  python src/F_Tasks/test_classifier.py [parameters] <model_type> [model-specific-parameters]
  ```

## Citation
TO DO
## License
TO DO
