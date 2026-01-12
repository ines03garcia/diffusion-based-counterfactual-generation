"""
RePaint images using a trained DDPM model for inpainting.
"""
import torch
import pandas as pd
import cv2
from PIL import Image
import numpy as np
import argparse
import os
import datetime

from src.DDPM.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from src.DDPM.guided_diffusion import dist_util, logger
from src.config import LOGS_PATH, MODELS_ROOT, METADATA_ROOT, IMAGES_ROOT, MASKS_DIR, DATASET_DIR


# ---------------------------
# I/O helpers
# ---------------------------
def load_image(path):
    """Load and preprocess grayscale image to [0,1] range, no extra normalization."""
    img = Image.open(path).convert("L")
    img = np.array(img).astype(np.float32) / 255.0  # Only scale to [0,1]
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return img

def save_image(tensor, path):
    """Save tensor as grayscale image"""
    img = tensor.squeeze().detach().cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def load_mask(path, target_hw, invert=False, device="cpu"):
    H, W = target_hw
    m = Image.open(path).convert("L").resize((W, H), Image.NEAREST)
    m = np.array(m).astype(np.float32) / 255.0
    m = (m > 0.5).astype(np.float32)  # binarize
    if invert:
        m = 1.0 - m
    m = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device)
    m = m.clamp(0.0, 1.0)
    return m  # 1 = keep, 0 = inpaint


def draw_bbox(image_path, example_name, image_with_bbox_path):
    metadata = os.path.join(METADATA_ROOT, "resized_annotations_512.csv")
    df = pd.read_csv(metadata)

    # Find the row for this image
    image_row = df[df['image_id'] == example_name]
    
    if image_row.empty:
        logger.log(f"No annotations found for image: {example_name}")
        return

    for col in ['resized_xmin', 'resized_ymin', 'resized_xmax', 'resized_ymax']:
        image_row[col] = image_row[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

    row = image_row.iloc[0]
    
    # Load as tensor to maintain consistency with save_image function
    img_tensor = load_image(image_path)
    
    # Convert tensor to numpy for cv2 operations
    img_np = img_tensor.squeeze().detach().cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)  # Convert to 0-255 range
    
    nr_anomalies = len(row['resized_xmin'])
    for i in range(nr_anomalies):
        x_min = int(row['resized_xmin'][i])
        y_min = int(row['resized_ymin'][i])
        x_max = int(row['resized_xmax'][i])
        y_max = int(row['resized_ymax'][i])
        
        # Draw rectangle in black (0 value)
        cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), 0, 2)
    
    # Convert back to tensor for consistent saving
    img_with_bbox = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    
    # Save using the same function to ensure consistency
    save_image(img_with_bbox, image_with_bbox_path)
    logger.log(f"Saved result with bounding boxes: {image_with_bbox_path}")

    

# ---------------------------
# Model loading
# ---------------------------
def load_model(args, device):
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)

    if args.model_path:
        logger.log(f"loading model from checkpoint: {args.model_path}")
        state_dict = dist_util.load_state_dict(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.log("model loaded successfully")

    model.eval()
    return model, diffusion


# ---------------------------
# RePaint logic
# ---------------------------
def project_known_region(diffusion, x0, x_t, t_tensor, mask, noise=None):
    """
    RePaint projection step:
      Replace KEEP region (mask==1) in x_t with the correctly noised
      version of the original image at timestep t: q(x_t | x0).
      Inpaint region (mask==0) is left unchanged.
    """
    x0_to_t = diffusion.q_sample(x_start=x0, t=t_tensor, noise=noise)
    return mask * x0_to_t + (1.0 - mask) * x_t


def repaint_inpaint(args, model, diffusion, device, example_name):
    """
    Main RePaint inpainting function with jumping mechanism.
    
    The RePaint paper introduces a jumping mechanism where every jump_length iterations,
    the scheduler jumps back by jump_n_sample steps and then continues forward.
    This helps improve the harmony between the inpainted and known regions by allowing
    the model to reconsider previous decisions with updated context.
    """
    logger.log(f"[RePaint] image: {args.image_path}")
    x0 = load_image(args.image_path).to(device)
    H, W = x0.shape[-2], x0.shape[-1]

    output_dir = getattr(args, "example_output_dir", args.images_dir)
    os.makedirs(output_dir, exist_ok=True)

    #save_image(x0, os.path.join(output_dir, "input.png"))

    if args.mask_path is None or not os.path.exists(args.mask_path):
        print("RePaint requires --mask_path pointing to a binary mask.")
        return
        #raise FileNotFoundError("RePaint requires --mask_path pointing to a binary mask.")
    mask = load_mask(args.mask_path, (H, W), invert=args.mask_invert, device=device)
    #save_image(mask, os.path.join(output_dir, "mask.png"))

    # Visualize input with bounding boxes
    #draw_bbox(args.image_path, example_name, os.path.join(output_dir, "input_with_bbox.png"))

    # Start from noise, project known region at T
    with torch.no_grad():
        T = args.diffusion_steps - 1
        t = torch.tensor([T], device=device)
        x_t = torch.randn_like(x0)
        known_noise = torch.randn_like(x0)
        x_t = project_known_region(diffusion, x0, x_t, t, mask, noise=known_noise)

        logger.log(f"[RePaint] reverse diffusion with {args.diffusion_steps} steps and jumping every {args.jump_length} steps...")
        
        # Initialize jump parameters
        jump_length = getattr(args, 'jump_length', 10)  # Jump every 10 iterations
        jump_n_sample = getattr(args, 'jump_n_sample', 10)  # Number of steps to jump back
        
        current_step = args.diffusion_steps - 1
        iteration_count = 0
        
        while current_step >= 0:
            t = torch.tensor([current_step], device=device)

            # Project known region at current t
            x_t = project_known_region(diffusion, x0, x_t, t, mask, noise=known_noise)

            # One denoising step
            out = diffusion.p_sample(model, x_t, t)
            x_tm1 = out["sample"]

            # Project at t-1 if not at the final step
            if current_step > 0:
                t_prev = torch.tensor([current_step - 1], device=device)
                x_tm1 = project_known_region(diffusion, x0, x_tm1, t_prev, mask, noise=known_noise)

            x_t = x_tm1
            iteration_count += 1
            
            # Move to next step
            current_step -= 1

            # RePaint jumping mechanism: every jump_length iterations, jump back
            if jump_length > 0 and iteration_count % jump_length == 0 and current_step >= 0:
                # Jump back by jump_n_sample steps (but not beyond the starting point)
                jump_back_to = min(current_step + jump_n_sample, args.diffusion_steps - 1)
                
                if jump_back_to > current_step:
                    logger.log(f"[RePaint] Jumping back from step {current_step} to step {jump_back_to}")
                    
                    # Jump back by adding noise
                    t_jump_back = torch.tensor([jump_back_to], device=device)
                    t_current = torch.tensor([current_step], device=device)
                    
                    # Calculate the number of steps to jump back
                    steps_back = jump_back_to - current_step
                    
                    # Add noise to simulate going back in time
                    jump_noise = torch.randn_like(x_t)
                    
                    # Use the forward process to add noise (q_sample)
                    # We go from current denoised state back to a more noisy state
                    x_t = diffusion.q_sample(x_start=x_t, t=torch.tensor([steps_back], device=device), noise=jump_noise)
                    
                    # Project the known region at the jumped-back timestep
                    x_t = project_known_region(diffusion, x0, x_t, t_jump_back, mask, noise=known_noise)
                    
                    # Update current step to the jumped-back position
                    current_step = jump_back_to

            
            # Save intermediate results
            #if (args.diffusion_steps - 1 - current_step) % 100 == 0:
                #save_image(x_t, os.path.join(output_dir, f"repaint_t#{current_step:04d}.png"))
        
        # Save final result
        final_img = x_t * (1 - mask) + x0 * mask

        #final_path = os.path.join(output_dir, "repaint_result.png")
        #save_image(final_img, final_path)
        #logger.log(f"[RePaint] saved: {final_path}")

        final_general_path = os.path.join(IMAGES_ROOT, f"counterfactuals/{example_name}")
        save_image(final_img, final_general_path)

        # Visualize output with bounding boxes
        #draw_bbox(final_path, example_name, os.path.join(output_dir, "output_with_bbox.png"))


# ---------------------------
# CLI
# ---------------------------
def main():
    args = create_argparser().parse_args()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")
    args.output_dir = os.path.join(LOGS_PATH, f"RePaint_logs/job_{slurm_job_id}_{timestamp}")
    args.images_dir = os.path.join(args.output_dir, "images")

    device = dist_util.dev()
    dist_util.setup_dist()
    logger.configure(experiment_type="repaint")

    model, diffusion = load_model(args, device)
    os.makedirs(os.path.join(IMAGES_ROOT, "counterfactuals"), exist_ok=True)

    for i, example_name in enumerate(args.examples):
        
        if example_name in os.listdir(os.path.join(IMAGES_ROOT,"counterfactuals")):
            logger.log(f"[RePaint] counterfactual already created for: {example_name}")
            continue
        
        args.image_path = os.path.join(DATASET_DIR, example_name)
        args.mask_path = os.path.join(MASKS_DIR, example_name)
        
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"File not found: {args.image_path}")
        
        args.example_output_dir = os.path.join(
            args.images_dir, f"example_{i+1:02d}_{os.path.splitext(example_name)[0]}"
        )

        repaint_inpaint(args, model, diffusion, device, example_name)


def create_argparser():
    
    metadata = os.path.join(METADATA_ROOT, "resized_annotations_512.csv")
    df = pd.read_csv(metadata)
    anomalous_images = df['image_id'].tolist()

    defaults = dict(
        image_size=512,
        in_channels=1,
        out_channels=1,
        num_channels=256,
        num_res_blocks=3,
        attention_resolutions="32,16",
        num_heads=-1,
        dropout=0.0,
        resblock_updown=True,
        use_checkpoint=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        model_path=os.path.join(MODELS_ROOT, "model009000.pt"),
        examples=anomalous_images,
        mask_invert=False,
        use_fp16=False,
        jump_length=-1,
        jump_n_sample=-1
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()