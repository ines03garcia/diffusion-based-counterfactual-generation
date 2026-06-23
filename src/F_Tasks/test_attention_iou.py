import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
METADATA_ROOT = os.path.join(PROJECT_ROOT, "data/metadata")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data/images")

from src.E_Aux_Scripts.utils import set_seed, seed_worker
from src.E_Aux_Scripts.classifier_helpers import (
    build_model,
    create_transforms,
    get_test_dataset,
    load_checkpoint_weights,
)
from src.E_Aux_Scripts.argument_parsers import create_test_argparser

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget


def make_reshape_transform_vit(model):
    vit = model.vit if hasattr(model, "vit") else model
    grid_h, grid_w = vit.patch_embed.grid_size

    def reshape_transform(tensor):
        # tensor: [B, tokens, C]
        B, N, C = tensor.shape

        num_patches = grid_h * grid_w

        if N == num_patches + 1:
            tensor = tensor[:, 1:, :]  # remove CLS token
        elif N != num_patches:
            raise RuntimeError(
                f"Unexpected token count: got {N}, expected {num_patches} "
                f"or {num_patches + 1}. grid_size={(grid_h, grid_w)}"
            )

        return tensor.reshape(B, grid_h, grid_w, C).permute(0, 3, 1, 2)

    return reshape_transform


def get_target_layers(model, model_type):
    model_type = model_type.lower()

    if model_type == "vit":
        # Your wrapper: VisionTransformerClassifier.vit is the timm ViT
        if hasattr(model, "vit") and hasattr(model.vit, "blocks"):
            return [model.vit.blocks[-1].norm1], make_reshape_transform_vit(model)

        # Direct timm ViT
        if hasattr(model, "blocks"):
            return [model.blocks[-1].norm1], make_reshape_transform_vit(model)

        raise AttributeError("Could not find ViT blocks. Expected model.vit.blocks or model.blocks.")

    raise ValueError(f"Unsupported model_type for Grad-CAM: {model_type}")


class ModelGradCAM:
    def __init__(self, model, model_type):
        target_layers, reshape_transform = get_target_layers(model, model_type)

        self.cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

    def __call__(self, images, target_class=1):
        targets = [BinaryClassifierOutputTarget(target_class)] * images.size(0)

        cam = self.cam(
            input_tensor=images,
            targets=targets,
        )

        return torch.from_numpy(cam).to(images.device)
    
def load_metadata(path):
    with open(path, "r") as f:
        rows = json.load(f)
    return {r["image_id"]: r for r in rows}


def get_boxes(row):
    keys = ["resized_xmin", "resized_ymin", "resized_xmax", "resized_ymax"]
    if not all(k in row for k in keys):
        return []

    boxes = []
    for x1, y1, x2, y2 in zip(row["resized_xmin"], row["resized_ymin"],
                              row["resized_xmax"], row["resized_ymax"]):
        if x2 > x1 and y2 > y1:
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def boxes_to_mask(boxes, h, w, device):
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        mask[y1:y2, x1:x2] = True
    return mask


def compute_iou(pred_mask, roi_mask):
    inter = (pred_mask & roi_mask).sum().float()
    union = (pred_mask | roi_mask).sum().float()
    return float(inter / union) if union > 0 else None


def main():
    parser = create_test_argparser()
    parser.add_argument("--cam_quantile", type=float, default=0.8)
    parser.add_argument("--iou_output", type=str, default="gradcam_iou.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_dir is None:
        if args.dataset.lower() == "vindr":
            args.data_dir = os.path.join(IMAGES_ROOT, "VinDrMammo-CLIP-CLAHE")
        elif args.dataset.lower() == "inbreast":
            args.data_dir = os.path.join(IMAGES_ROOT, "datasets/Inbreast_png")
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.metadata_path is None:
        if args.dataset.lower() == "vindr":
            args.metadata_path = os.path.join(METADATA_ROOT, "processed_df_birads.json")
        elif args.dataset.lower() == "inbreast":
            args.metadata_path = os.path.join(METADATA_ROOT, "inbreast_test_metadata.csv")
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

    metadata = load_metadata(args.metadata_path)

    _, test_transform = create_transforms(
        augmentation_type="none",
        model_type=args.model_type,
    )

    test_dataset = get_test_dataset(args, test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=lambda x: seed_worker(args, x),
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = build_model(args, device, experiment="test")

    if args.model_type.lower() == "vit":
        vit = model.vit if hasattr(model, "vit") else model
        print("ViT patch grid:", vit.patch_embed.grid_size)
        print("ViT image size:", vit.patch_embed.img_size)
        print("ViT patch size:", vit.patch_embed.patch_size)

    ckpt = args.checkpoint_path or os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(MODELS_ROOT, ckpt)

    load_checkpoint_weights(model, ckpt, device)
    model.eval()

    cam_extractor = ModelGradCAM(model, args.model_type)

    results = []

    for batch in tqdm(test_loader, desc="Computing Grad-CAM IoU"):
        images, targets, image_ids = batch[:3]
        images = images.to(device)

        cam_maps = cam_extractor(images)  # [B, H, W]

        flat = cam_maps.flatten(1)
        thr = torch.quantile(flat, args.cam_quantile, dim=1).view(-1, 1, 1)
        cam_masks = cam_maps >= thr

        for i, image_id in enumerate(image_ids):
            image_id = str(image_id)

            row = metadata.get(image_id)
            boxes = get_boxes(row) if row is not None else []

            if not boxes:
                continue

            roi_mask = boxes_to_mask(
                boxes,
                images.shape[-2],
                images.shape[-1],
                images.device,
            )

            iou = compute_iou(cam_masks[i], roi_mask)

            results.append({
                "image_id": image_id,
                "target": int(targets[i]),
                "num_boxes": len(boxes),
                "iou": iou,
            })
        break

    valid_ious = [r["iou"] for r in results if r["iou"] is not None]

    output = {
        "checkpoint": ckpt,
        "dataset": args.dataset,
        "model_type": args.model_type,
        "cam_quantile": args.cam_quantile,
        "num_images_with_roi": len(valid_ious),
        "mean_iou": float(np.mean(valid_ious)) if valid_ious else None,
        "std_iou": float(np.std(valid_ious)) if valid_ious else None,
        "per_image": results,
    }

    with open(args.iou_output, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(
        {k: output[k] for k in output if k != "per_image"},
        indent=2,
    ))


if __name__ == "__main__":
    main()