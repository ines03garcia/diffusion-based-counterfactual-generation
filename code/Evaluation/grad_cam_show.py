from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import sys
import os
import cv2
import numpy as np

# Access convnext, vit and dataset modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../Classifiers/scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../Classifiers/aux_scripts'))
from convNeXt import ConvNeXtClassifier
from vision_transformer import VisionTransformerClassifier, create_transforms
from aux_scripts.config import DATA_DIR, DATA_ROOT, METADATA_ROOT
from aux_scripts.VinDrMammo_dataset import VinDrMammo_dataset


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

# Load test split of the dataset
metadata_csv = os.path.join(METADATA_ROOT, "resized_df_counterfactuals.csv")
_, val_transform = create_transforms("none") # No augmentation

anomalous_with_findings_test_dataset = VinDrMammo_dataset(
    data_dir=DATA_ROOT,
    metadata_path=metadata_csv,
    split="test",
    transform=val_transform,
    testing_category="counterfactuals_only",
    testing_cf = True,
    counterfactuals_dir = os.path.join(DATA_DIR, "counterfactuals_512")
)

print(f"Loaded test split: {len(anomalous_with_findings_test_dataset)} images")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
targets = [ClassifierOutputTarget(0)] # Show GradCAM for class 0 (healthy)

for model_type in ['convnext', 'vit']:
    for checkpoint_type in ['no_cf', 'cf']:
        checkpoint_path = f'/home/inescgarcia/Documents/BolsaInvestigacao/diffusion-counterfactual-generation/models/{model_type}_{checkpoint_type}.pth'
        
        model = model_load(checkpoint_path, model_type, device)

        # Select the target layer for GradCAM using correct attribute names
        if model_type == 'convnext':
            last_cnblock = model.convnext.features[-1][-1]
            # Get the Conv2d inside the CNBlock
            target_layers = [last_cnblock.block[0]]
        elif model_type == 'vit':
            print(model.vit.encoder.layers[-1].mlp)
            target_layers = [model.vit.encoder.layers[-1].mlp]

        print(f"Using model: {model_type} with checkpoint: {checkpoint_type}")
        healthy_ious = []
        nonhealthy_ious = []
        
        # Iterate over test dataset and compute GradCAM for each image
        for idx in range(len(anomalous_with_findings_test_dataset)):
            img, label, img_name = anomalous_with_findings_test_dataset[idx]

            # Preprocess image matching model's training and convert tensor to numpy image for GradCAM visualization
            img_np = img.cpu().numpy()
            rgb_img = np.transpose(img_np, (1, 2, 0))
            rgb_img = np.clip((rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])), 0, 1)

            input_tensor = img.unsqueeze(0).to(device)
            print(f"Processing image: {img_name} with label: {label}")

            out = model(input_tensor)

            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                print("GradCAM computed.")
                if grayscale_cam.ndim == 3:
                    print("CAM output shape (batch_size, H, W):", grayscale_cam.shape)
                    grayscale_cam = grayscale_cam[0, :] # Convert to (H, W)
                
                grayscale_cam = 1.0 - grayscale_cam

                mask_path = os.path.join(DATA_DIR, "masks_512", img_name)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 0).astype(np.float32)  # Convert to binary mask

                    # Resize mask to match CAM shape if needed
                    if mask.shape != grayscale_cam.shape:
                        print(f"Resizing mask from {mask.shape} to {grayscale_cam.shape}")
                        mask = cv2.resize(mask, (grayscale_cam.shape[1], grayscale_cam.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Quantify overlay: e.g., sum of CAM values inside mask, IoU, etc.
                    cam_in_mask = np.sum(grayscale_cam * mask)
                    mask_area = np.sum(mask)
                    cam_total = np.sum(grayscale_cam)
                    iou = np.sum((grayscale_cam > 0.5) * mask) / np.sum(((grayscale_cam > 0.5) + mask) > 0)

                    print(f"Image: {img_name}, CAM in mask: {cam_in_mask:.4f}, Mask area: {mask_area}, CAM total: {cam_total:.4f}, IoU: {iou:.4f}")

                    rgb_img = np.float32(rgb_img)
                    grayscale_cam = np.float32(grayscale_cam)
                    if grayscale_cam.shape != rgb_img.shape[:2]:
                        grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
                        
                    # function receives RGB or BGR image
                    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    output_dir = f'../../data/images/gradcam/{model_type}_{checkpoint_type}'
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(f'{output_dir}/{img_name}', np.uint8(255 * visualization))
                else:
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")


# Given heatmaps from Grad-CAM for different checkpoints (with and without counterfactual training), quantify overlay between heatmaps and ground truth lesion masks.