"""
    Given heatmaps from Grad-CAM for different checkpoints (with and without counterfactual training), quantify overlay between heatmaps and ground truth lesion masks.
""" 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
#from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import os
import cv2
import numpy as np
import math
from PIL import Image

from src.Classifiers.aux_scripts.ClassifierConvNeXt import ConvNeXtClassifier
from src.Classifiers.aux_scripts.ClassifierVisionTransformer import VisionTransformerClassifier
from src.config import DATASET_DIR, MASKS_DIR, IMAGES_ROOT, METADATA_ROOT, DATA_ROOT, MODELS_ROOT
from src.Classifiers.aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset
from src.Classifiers.aux_scripts.utils import create_transforms

# Custom target for positive class (class 1)
class PositiveLogitTarget:
    def __call__(self, model_output):
        if model_output.dim() == 1:
            return model_output[0]
        elif model_output.dim() == 2:
            return model_output[:, 0]
        raise ValueError(f"Unexpected model_output shape: {model_output.shape}")
        
# Custom target for negative class (class 0)
class NegativeLogitTarget:
    def __call__(self, model_output):
        # Handle both squeezed [batch] and unsqueezed [batch, 1] formats
        if model_output.dim() == 1:
            return -model_output[0]
        elif model_output.dim() == 2:
            return -model_output[:, 0]
        raise ValueError(f"Unexpected model_output shape: {model_output.shape}")

def model_load(checkpoint_path, model_type, device):
    if model_type.lower() == "vit":
        model = VisionTransformerClassifier(num_classes=1, pretrained=False, grad_cam=True)
    elif model_type.lower() == "convnext":
        model = ConvNeXtClassifier(num_classes=1, pretrained=False, grad_cam=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'vit' or 'convnext'")
    
    state = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model

def overlay_cam_on_image(img, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    cam = np.uint8(255 * cam) # Scale to [0, 255] and convert to uint8 as required by applyColorMap
    heatmap = cv2.applyColorMap(cam, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0 # Convert BGR to RGB and scale to [0, 1]
    overlay = heatmap * alpha + img * (1 - alpha) # Alpha is the transparency factor
    overlay = np.clip(overlay, 0, 1) # Ensure values are in [0, 1]
    return overlay

_, val_transform = create_transforms("none") # No augmentation

anomalous_with_findings_test_dataset = VinDrMammo_dataset(
    data_dir=DATASET_DIR,
    metadata_path=os.path.join(METADATA_ROOT, "resized_df_counterfactuals.csv"),
    split="test",
    transform=val_transform,
    testing_category="anomalous_with_findings",
    testing_cf = False,
    counterfactuals_dir = os.path.join(IMAGES_ROOT, "repaint_results")
)

def reshape_transform(tensor, height=14, width=14):
    # tensor: [B, 1+N, D]  (CLS + patch tokens)
    # drop CLS token:
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # -> [B, H, W, D] then to [B, D, H, W]
    result = result.transpose(2, 3).transpose(1, 2)
    return result

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for model_type in ['ConvNeXt', 'ViT']:
    for checkpoint_type in ['_no_cf', '_cf']:
        checkpoint_path = os.path.join(MODELS_ROOT, f'{model_type}{checkpoint_type}.pth')
        model = model_load(checkpoint_path, model_type, device)

        # Select the target layer for GradCAM using correct attribute names
        if model_type.lower() == 'convnext':
            # CNBlock (before pooling/classifier)
            target_layers = [model.convnext.features[-1][-1]]

        elif model_type.lower() == 'vit':
            target_layers = [model.vit.encoder.layers.encoder_layer_11.ln_1]
        
        # IoU accumulators for healthy and non-healthy predictions
        healthy_ious = []
        nonhealthy_ious = []

        # Iterate over test dataset and compute GradCAM for each image
        for idx in range(len(anomalous_with_findings_test_dataset)):
            img, label, img_name = anomalous_with_findings_test_dataset[idx]
            img_np = img.cpu().numpy()
            rgb_img = np.transpose(img_np, (1, 2, 0))
            rgb_img = np.clip((rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])), 0, 1)

            input_tensor = img.unsqueeze(0).to(device)

            gradcam_args = {
                "model": model,
                "target_layers": target_layers
            }

            if model_type.lower() == 'vit':
                gradcam_args["reshape_transform"] = reshape_transform

            # Get model prediction for this image
            out = model(input_tensor)           # [B,1]
            logit = out[:, 0] if out.dim() == 2 else out
            prob = torch.sigmoid(logit).item()
            pred_class = 1 if prob >= 0.5 else 0

            targets = [PositiveLogitTarget()] if pred_class == 1 else [NegativeLogitTarget()]

            with GradCAM(**gradcam_args) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
                if grayscale_cam.ndim != 2:
                    raise ValueError(f"Expected 2D CAM, got {grayscale_cam.shape}")
                if np.isnan(grayscale_cam).any() or np.isinf(grayscale_cam).any():
                    raise ValueError("CAM contains NaN/Inf.")
                if grayscale_cam.max() <= 1e-8:
                    print("Warning: CAM max is ~0. This can happen if gradients are zero.")

                if grayscale_cam.ndim == 3:
                    grayscale_cam = grayscale_cam[0, :] # (224, 224)

                mask_path = os.path.join(MASKS_DIR, img_name)

                if os.path.exists(mask_path):
                    # Load mask using PIL for consistency
                    mask = np.array(Image.open(mask_path).convert('L'))
                    mask = (mask == 0).astype(np.float32)  # Black pixels (0) = ROI

                    if mask.shape != grayscale_cam.shape: # Resize masks from 512 to 224
                        # Resize using PIL
                        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_pil = mask_pil.resize((grayscale_cam.shape[1], grayscale_cam.shape[0]), Image.NEAREST)
                        mask = np.array(mask_pil).astype(np.float32) / 255.0 # [0, 1]

                    # Percentile threshold
                    threshold = np.percentile(grayscale_cam, 75)
                    
                    # Binarize CAM based on threshold
                    cam_binary = (grayscale_cam >= threshold).astype(np.float32)
                    
                    # Calculate IoU
                    intersection = np.sum(cam_binary * mask)
                    union = np.sum((cam_binary + mask) > 0)
                    iou = intersection / union if union > 0 else 0.0
                    
                    # Additional metrics for debugging
                    cam_in_mask = np.sum(grayscale_cam * mask)
                    mask_area = np.sum(mask)
                    cam_total = np.sum(grayscale_cam)
                    cam_max = np.max(grayscale_cam)
                    cam_mean = np.mean(grayscale_cam)

                    # Accumulate IoU based on prediction
                    if pred_class == 0:
                        healthy_ious.append(iou)
                    else:
                        nonhealthy_ious.append(iou)
                else:
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")

                # Save visualization regardless of mask existence
                visualization = overlay_cam_on_image(rgb_img, grayscale_cam)
                gradcam_images_dir = os.path.join(IMAGES_ROOT, "gradcam2")
                os.makedirs(gradcam_images_dir, exist_ok=True)
                output_dir = f'{gradcam_images_dir}/{model_type}{checkpoint_type}'
                os.makedirs(output_dir, exist_ok=True)
                # Save using PIL for consistency
                vis_img = (np.clip(visualization, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(vis_img).save(f'{output_dir}/{img_name}')
                
        # Calculate and log global IoU and prediction counts
        avg_healthy = np.mean(healthy_ious) if healthy_ious else float('nan')
        avg_nonhealthy = np.mean(nonhealthy_ious) if nonhealthy_ious else float('nan')
        all_ious = healthy_ious + nonhealthy_ious
        global_iou = np.mean(all_ious) if all_ious else float('nan')
        n_healthy = len(healthy_ious)
        n_nonhealthy = len(nonhealthy_ious)
        
        log_dir = os.path.join(DATA_ROOT, 'logs/gradcam_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{model_type}{checkpoint_type}_iou.txt')
        
        with open(log_path, 'w') as f:
            f.write(f'Model: {model_type}\n')
            f.write(f'Checkpoint: {checkpoint_type}\n')
            f.write(f'Average IoU for healthy predictions: {avg_healthy:.4f}\n')
            f.write(f'Average IoU for non-healthy predictions: {avg_nonhealthy:.4f}\n')
            f.write(f'Global IoU (all predictions): {global_iou:.4f}\n')
            f.write(f'Number of healthy predictions: {n_healthy}\n')
            f.write(f'Number of non-healthy predictions: {n_nonhealthy}\n')