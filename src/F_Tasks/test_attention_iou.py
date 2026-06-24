import os, sys, json
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
METADATA_ROOT = os.path.join(PROJECT_ROOT, "data/metadata")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data/images")
LOGS_ROOT = os.path.join(PROJECT_ROOT, "data/logs")

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


def get_dataset_image_id(dataset, idx):
    if hasattr(dataset, "image_ids"):
        return dataset.image_ids[idx]
    if hasattr(dataset, "image_names"):
        return dataset.image_names[idx]
    return dataset[idx][2]


def filter_dataset_with_boxes(dataset, metadata):
    keep_indices = []
    skipped_no_metadata = 0
    skipped_no_boxes = 0

    for idx in range(len(dataset)):
        image_id = str(get_dataset_image_id(dataset, idx))
        row = metadata.get(image_id)
        if row is None:
            skipped_no_metadata += 1
            continue

        if not get_boxes(row):
            skipped_no_boxes += 1
            continue

        keep_indices.append(idx)

    return Subset(dataset, keep_indices), {
        "dataset_images_before_roi_filter": len(dataset),
        "dataset_images_after_roi_filter": len(keep_indices),
        "prefilter_skipped_no_metadata": skipped_no_metadata,
        "prefilter_skipped_no_boxes": skipped_no_boxes,
    }


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


def compute_box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def compute_max_box_iou(pred_box, gt_boxes):
    if pred_box is None or not gt_boxes:
        return 0.0
    return max(compute_box_iou(pred_box, gt_box) for gt_box in gt_boxes)


def largest_connected_component_bbox(mask):
    if not np.any(mask):
        return None

    from scipy import ndimage

    label_im, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return None

    sizes = ndimage.sum(mask, label_im, range(1, num_labels + 1))
    largest_label = int(np.argmax(sizes)) + 1
    ys, xs = np.where(label_im == largest_label)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    )


def binarize_cam_maps(cam_maps, quantile):
    flat = cam_maps.flatten(1)
    thr = torch.quantile(flat, quantile, dim=1).view(-1, 1, 1)
    return (cam_maps > thr) & (cam_maps > 0), thr


def sanitize_filename(value):
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(value))


def denormalize_image(image, args):
    image = image.detach().cpu().float()

    if args.model_type in ["mammo-clip", "fpn-mil"]:
        mean = torch.tensor([getattr(args, "mean", 0.400409)] * 3).view(3, 1, 1)
        std = torch.tensor([getattr(args, "std", 0.259367)] * 3).view(3, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def colorize_cam(cam):
    try:
        import matplotlib.pyplot as plt

        return plt.get_cmap("jet")(cam)[..., :3]
    except Exception:
        cam = np.clip(cam, 0, 1)
        return np.stack([cam, np.sqrt(cam), 1.0 - cam], axis=-1)


def draw_mask_edges(draw, mask, color):
    if not np.any(mask):
        return

    h, w = mask.shape
    edge = np.zeros_like(mask, dtype=bool)
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge &= mask

    ys, xs = np.where(edge)
    for x, y in zip(xs, ys):
        if 0 <= x < w and 0 <= y < h:
            draw.point((int(x), int(y)), fill=color)


def make_case_visualization(case):
    image = case["image"]
    cam = np.clip(case["cam"], 0, 1)
    cam_mask = case["cam_mask"]
    roi_mask = case["roi_mask"]
    boxes = case["boxes"]

    overlay = np.clip(0.55 * image + 0.45 * colorize_cam(cam), 0, 1)
    canvas = Image.fromarray((overlay * 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)

    for x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=4)

    if case["method"] == "boundingbox" and case.get("pred_box") is not None:
        x1, y1, x2, y2 = case["pred_box"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=4)
    else:
        draw_mask_edges(draw, cam_mask, (255, 255, 0))
        draw_mask_edges(draw, roi_mask, (0, 255, 0))

    caption_lines = [
        f"image_id: {case['image_id']}",
        f"method: {case['method']}",
        f"IoU (%): {case['iou'] * 100:.2f}",
    ]
    font_size = max(18, int(canvas.height * 0.022))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in caption_lines]
    line_heights = [bbox[3] - bbox[1] for bbox in line_bboxes]
    box_width = min(
        canvas.width,
        max(bbox[2] - bbox[0] for bbox in line_bboxes) + 32,
    )
    box_height = sum(line_heights) + 36 + 8 * (len(caption_lines) - 1)
    draw.rectangle((0, 0, box_width, box_height), fill=(0, 0, 0))

    y = 16
    for line, line_height in zip(caption_lines, line_heights):
        draw.text((16, y), line, fill=(255, 255, 255), font=font)
        y += line_height + 8

    return canvas


def update_best_cases(best_cases, case, max_cases):
    if max_cases <= 0 or case["iou"] is None:
        return

    best_cases.append(case)
    best_cases.sort(key=lambda item: item["iou"], reverse=True)
    del best_cases[max_cases:]


def save_best_visualizations(best_cases, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    panels = []

    for rank, case in enumerate(best_cases, start=1):
        panel = make_case_visualization(case)
        filename = f"rank_{rank}_{sanitize_filename(case['image_id'])}_iou_{case['iou']:.4f}.png"
        path = os.path.join(output_dir, filename)
        panel.save(path)
        saved_paths.append(path)
        panels.append(panel)

    if panels:
        gap = 12
        width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
        height = max(panel.height for panel in panels)
        grid = Image.new("RGB", (width, height), (0, 0, 0))

        x_offset = 0
        for panel in panels:
            grid.paste(panel, (x_offset, 0))
            x_offset += panel.width + gap

        grid_path = os.path.join(output_dir, "best_3_cases.png")
        grid.save(grid_path)
        saved_paths.append(grid_path)

    return saved_paths


def resolve_output_paths(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or os.path.join(LOGS_ROOT, "attention_iou", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    iou_output = args.iou_output or "gradcam_iou.json"
    if not os.path.isabs(iou_output):
        iou_output = os.path.join(log_dir, iou_output)

    return log_dir, iou_output, os.path.join(log_dir, "visualizations")


def main():
    parser = create_test_argparser()
    parser.add_argument("--cam_quantile", type=float, default=0.8)
    parser.add_argument("--method", type=str, choices=["pixel", "boundingbox"], default="pixel")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--iou_output", type=str, default=None)
    parser.add_argument("--num_visualizations", type=int, default=3)
    parser.add_argument("--max_batches", type=int, default=None)
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
    test_dataset, roi_filter_stats = filter_dataset_with_boxes(test_dataset, metadata)

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
    best_cases = []
    log_dir, iou_output, vis_dir = resolve_output_paths(args)
    batches_processed = 0
    images_seen = 0
    skipped_no_metadata = 0
    skipped_no_boxes = 0

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Computing Grad-CAM IoU")):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        batches_processed += 1
        images, targets, image_ids = batch[:3]
        images_seen += len(image_ids)
        images = images.to(device)

        cam_maps = cam_extractor(images)  # [B, H, W]

        cam_masks, _ = binarize_cam_maps(cam_maps, args.cam_quantile)

        for i, image_id in enumerate(image_ids):
            image_id = str(image_id)

            row = metadata.get(image_id)
            if row is None:
                skipped_no_metadata += 1
                continue

            boxes = get_boxes(row)
            if not boxes:
                skipped_no_boxes += 1
                continue

            roi_mask = boxes_to_mask(
                boxes,
                images.shape[-2],
                images.shape[-1],
                images.device,
            )

            pred_box = None
            if args.method == "pixel":
                iou = compute_iou(cam_masks[i], roi_mask)
            else:
                pred_box = largest_connected_component_bbox(
                    cam_masks[i].detach().cpu().numpy().astype(bool)
                )
                iou = compute_max_box_iou(pred_box, boxes)

            result = {
                "image_id": image_id,
                "target": int(targets[i]),
                "num_boxes": len(boxes),
                "method": args.method,
                "pred_box": list(pred_box) if pred_box is not None else None,
                "iou": iou,
            }
            results.append(result)

            if iou is not None:
                update_best_cases(
                    best_cases,
                    {
                        "image_id": image_id,
                        "target": int(targets[i]),
                        "num_boxes": len(boxes),
                        "method": args.method,
                        "pred_box": pred_box,
                        "iou": iou,
                        "boxes": boxes,
                        "image": denormalize_image(images[i], args),
                        "cam": cam_maps[i].detach().cpu().numpy(),
                        "cam_mask": cam_masks[i].detach().cpu().numpy().astype(bool),
                        "roi_mask": roi_mask.detach().cpu().numpy().astype(bool),
                    },
                    args.num_visualizations,
                )

    valid_ious = [r["iou"] for r in results if r["iou"] is not None]
    visualization_paths = save_best_visualizations(best_cases, vis_dir)

    output = {
        "checkpoint": ckpt,
        "dataset": args.dataset,
        "model_type": args.model_type,
        "cam_quantile": args.cam_quantile,
        "method": args.method,
        "log_dir": log_dir,
        "iou_output": iou_output,
        "summary_output": os.path.join(log_dir, "summary.txt"),
        "visualization_dir": vis_dir,
        "visualizations": visualization_paths,
        **roi_filter_stats,
        "max_batches": args.max_batches,
        "batches_processed": batches_processed,
        "images_seen": images_seen,
        "skipped_no_metadata": skipped_no_metadata,
        "skipped_no_boxes": skipped_no_boxes,
        "num_images_with_roi": len(valid_ious),
        "mean_iou": float(np.mean(valid_ious)) if valid_ious else None,
        "std_iou": float(np.std(valid_ious)) if valid_ious else None,
        "per_image": results,
    }

    with open(iou_output, "w") as f:
        json.dump(output, f, indent=2)

    summary_path = os.path.join(log_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {ckpt}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"CAM quantile: {args.cam_quantile}\n")
        f.write(f"Dataset images before ROI filter: {roi_filter_stats['dataset_images_before_roi_filter']}\n")
        f.write(f"Dataset images after ROI filter: {roi_filter_stats['dataset_images_after_roi_filter']}\n")
        f.write(f"Prefilter skipped no metadata: {roi_filter_stats['prefilter_skipped_no_metadata']}\n")
        f.write(f"Prefilter skipped no boxes: {roi_filter_stats['prefilter_skipped_no_boxes']}\n")
        f.write(f"Max batches: {args.max_batches}\n")
        f.write(f"Batches processed: {batches_processed}\n")
        f.write(f"Images seen: {images_seen}\n")
        f.write(f"Skipped no metadata: {skipped_no_metadata}\n")
        f.write(f"Skipped no boxes: {skipped_no_boxes}\n")
        f.write(f"Images with ROI: {len(valid_ious)}\n")
        f.write(f"Mean IoU: {output['mean_iou']}\n")
        f.write(f"Std IoU: {output['std_iou']}\n")
        f.write(f"JSON output: {iou_output}\n")
        f.write(f"Visualization dir: {vis_dir}\n")
        for rank, case in enumerate(best_cases, start=1):
            f.write(f"Rank {rank}: {case['image_id']} IoU={case['iou']:.6f}\n")

    print(json.dumps(
        {k: output[k] for k in output if k != "per_image"},
        indent=2,
    ))


if __name__ == "__main__":
    main()