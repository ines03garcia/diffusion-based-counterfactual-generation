# Diffusion-Based Counterfactual Generation in Mammography

This repository provides code for generating counterfactual images using diffusion models, evaluating their quality, and classifying them using deep learning techniques. The project is organized for research in medical imaging, specifically mammography, but can be adapted for other domains.

## Project Structure

- **code/**
  - `config.py`, `__init__.py`: Main configuration and initialization files.
  - **Classifiers/**
    - `aux_scripts/`: Utilities for dataset preparation and splitting.
    - `scripts/`: Deep learning models (ConvNeXt, ViT), classification scripts, and evaluation tools.
  - **DDPM/**
    - `guided_diffusion/`: Core diffusion model implementation and utilities.
    - `scripts/`: Training, sampling, and noise manipulation scripts for DDPM.
  - **Evaluation/**
    - Scripts for evaluating counterfactuals (FID, Grad-CAM, image quality).
    - `aux_scripts/`: Additional utilities for evaluation and data preparation.

- **data/**
  - `images/`: Contains generated images, masks, patches, and example data.
  - `gradcam_logs/`: IOU logs for Grad-CAM evaluations.
  - `metadata/`: CSV files with annotations and dataset splits.
  - `zips/`: Compressed data files.

- **models/**
  - Pretrained model weights for classifiers and diffusion models.
  - `zips/`: Compressed model files.

## Key Features

- **Diffusion Models**: Implementation of DDPM and guided diffusion for counterfactual image generation.
- **Classification**: Deep learning classifiers (ConvNeXt, ViT) for evaluating generated images.
- **Evaluation**: FID score, Grad-CAM, and image quality assessment scripts.
- **Medical Imaging Focus**: Tools and scripts tailored for mammography datasets (VinDr-Mammo).

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

- **Train diffusion models**: See `code/DDPM/scripts/image_train.py`
- **Sample images**: See `code/DDPM/scripts/image_sample.py`
- **Classify images**: See `code/Classifiers/scripts/classify_counterfactuals.py`
- **Evaluate counterfactuals**: See `code/Evaluation/`

## Citation
TO DO
## License
TO DO
