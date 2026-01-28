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

from src.Classifiers.aux_scripts.ClassifierConvNeXt import ConvNeXtClassifier
from src.Classifiers.aux_scripts.ClassifierVisionTransformer import VisionTransformerClassifier
from src.config import DATASET_DIR, MASKS_DIR, IMAGES_ROOT, METADATA_ROOT, DATA_ROOT, MODELS_ROOT
from src.Classifiers.aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset
from src.Classifiers.aux_scripts.utils import create_transforms

# Custom target for negative class (class 0)
class NegativeLogitTarget:
    def __call__(self, model_output):
        return -model_output

def model_load(checkpoint_path, model_type, device):
    if model_type.lower() == "vit":
        model = VisionTransformerClassifier(num_classes=1, pretrained=False)
    elif model_type.lower() == "convnext":
        model = ConvNeXtClassifier(num_classes=1, pretrained=False)
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

def reshape_transform(x):
    # x is [B, N, C] for torchvision ViT
    x = x[:, 1:, :]                 # drop CLS -> [B, 196, C]
    B, N, C = x.shape
    H = W = int(math.sqrt(N))       # 14 for 224x224 with patch16
    assert H * W == N
    return x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

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

print(f"Loaded test split: {len(anomalous_with_findings_test_dataset)} images")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for model_type in ['ConvNeXt', 'ViT']:
    for checkpoint_type in ['_no_cf', '_cf']:

        checkpoint_path = os.path.join(MODELS_ROOT, f'{model_type}{checkpoint_type}.pth')
        model = model_load(checkpoint_path, model_type, device)

        model_type = model_type.lower()
        # Select the target layer for GradCAM using correct attribute names
        if model_type == 'convnext':
            last_cnblock = model.convnext.features[-1][-1]
            target_layers = [last_cnblock.block[0]]

        elif model_type == 'vit':
            target_layers = [model.vit.encoder.layers[-1].ln_1]

        print(f"Using model: {model_type} with checkpoint: {checkpoint_type}")

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
            print(f"Processing image: {img_name} with label: {label}")


            gradcam_args = {
                "model": model,
                "target_layers": target_layers
            }
            if model_type == 'vit':
                gradcam_args["reshape_transform"] = reshape_transform

            # Get model prediction for this image
            out = model(input_tensor)
            pred = out.item() if hasattr(out, 'item') else out.detach().cpu().numpy().squeeze()
            prob = torch.sigmoid(torch.tensor(pred)).item()
            pred_class = 1 if prob >= 0.5 else 0

            # Set GradCAM target to match predicted class
            if pred_class == 1:
                targets = [ClassifierOutputTarget(0)]  # highlight positive evidence
            else:
                targets = [NegativeLogitTarget()]      # highlight negative evidence

            with GradCAM(**gradcam_args) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                print("GradCAM computed.")

                if grayscale_cam.ndim == 3:
                    print("CAM output shape (batch_size, H, W):", grayscale_cam.shape)
                    grayscale_cam = grayscale_cam[0, :]

                grayscale_cam = 1.0 - grayscale_cam

                mask_path = os.path.join(MASKS_DIR, img_name)

                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 0).astype(np.float32)

                    if mask.shape != grayscale_cam.shape:
                        print(f"Resizing mask from {mask.shape} to {grayscale_cam.shape}")
                        mask = cv2.resize(mask, (grayscale_cam.shape[1], grayscale_cam.shape[0]), interpolation=cv2.INTER_NEAREST)

                    cam_in_mask = np.sum(grayscale_cam * mask)
                    mask_area = np.sum(mask)
                    cam_total = np.sum(grayscale_cam)
                    iou = np.sum((grayscale_cam > 0.5) * mask) / np.sum(((grayscale_cam > 0.5) + mask) > 0)
                    print(f"Image: {img_name}, CAM in mask: {cam_in_mask:.4f}, Mask area: {mask_area}, CAM total: {cam_total:.4f}, IoU: {iou:.4f}")

                    # Accumulate IoU based on prediction
                    if pred_class == 0:
                        healthy_ious.append(iou)
                    else:
                        nonhealthy_ious.append(iou)

                    visualization = overlay_cam_on_image(rgb_img, grayscale_cam)
                    output_dir = f'../../data/images/gradcam2/{model_type}{checkpoint_type}'
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(f'{output_dir}/{img_name}', np.uint8(255 * visualization))
                else:
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                
        # Calculate and log global IoU and prediction counts
        avg_healthy = np.mean(healthy_ious) if healthy_ious else float('nan')
        avg_nonhealthy = np.mean(nonhealthy_ious) if nonhealthy_ious else float('nan')
        all_ious = healthy_ious + nonhealthy_ious
        global_iou = np.mean(all_ious) if all_ious else float('nan')
        n_healthy = len(healthy_ious)
        n_nonhealthy = len(nonhealthy_ious)
        log_dir = os.path.join(DATA_ROOT, 'gradcam_logs')
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