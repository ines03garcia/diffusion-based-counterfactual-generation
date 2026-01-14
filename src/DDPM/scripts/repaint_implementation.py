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
from tqdm import trange

import time

from src.DDPM.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from src.DDPM.guided_diffusion import dist_util, logger
from src.config import LOGS_PATH, MODELS_ROOT, METADATA_ROOT, MASKS_DIR, DATASET_DIR, CF_DIR


# ---------------------------
# I/O helpers
# ---------------------------
def load_image(path):
    """Load and preprocess grayscale image to [0,1] range, no extra normalization."""
    img = Image.open(path).convert("L")
    img = np.array(img).astype(np.float32) / 255.0  # Scale to [0,1]
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return img

def save_image(tensor, path):
    """Save tensor as grayscale image"""
    img = tensor.squeeze().detach().cpu().numpy() # (H, W)
    img = np.clip(img * 255, 0, 255).astype(np.uint8) # Scale back to [0,255]
    Image.fromarray(img).save(path)


def load_mask(path, target_hw, device="cpu"):
    H, W = target_hw
    m = Image.open(path).convert("L").resize((W, H), Image.NEAREST)
    m = np.array(m).astype(np.float32) / 255.0 
    m = (m > 0.5).astype(np.float32)  # Binary mask
    m = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device) # (1,1,H,W)
    m = m.clamp(0.0, 1.0) # Do I need this ?
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
    if args.debugging:
        logger.log("Creating model and diffusion...")
    t0 = time.time()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    t1 = time.time()
    if args.debugging:
        logger.log(f"[Timing] Model and diffusion creation took {t1 - t0:.2f} seconds.")

    if args.model_path:
        if args.debugging:
            logger.log(f"Loading model from checkpoint: {args.model_path}")
        t2 = time.time()
        state_dict = dist_util.load_state_dict(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        t3 = time.time()
        if args.debugging:
            logger.log(f"[Timing] Model loading took {t3 - t2:.2f} seconds.")
        if args.debugging:
            logger.log("Model loaded successfully.")
    else:
        raise ValueError("Model path must be provided.")

    model.eval()
    return model, diffusion


# ---------------------------
# RePaint logic
# ---------------------------
def project_known_region(diffusion, x0, x_t, t_tensor, mask, noise=None):
    """
    RePaint projection step:
      Replace region to keep (mask==1) in x_t with the correct noisy
      version from the forward process at timestep t: q(x_t | x0).
      Inpainted region (mask==0) is added to the final result.
    """
    x0_to_t = diffusion.q_sample(x_start=x0, t=t_tensor, noise=noise)
    return mask * x0_to_t + (1.0 - mask) * x_t

def repaint_jump_step(x_t, current_step, jump_n_sample, diffusion_steps, diffusion, device, debugging, logger, x0, mask, known_noise):
    """
        Jump back by jump_n_sample steps (but not beyond the starting point)
        and project the known region at the jumped-back timestep.
    """
    jump_back_to = min(current_step + jump_n_sample, diffusion_steps - 1)
    
    if jump_back_to > current_step:
        if debugging:
            logger.log(f"[RePaint] Jumping back from step {current_step} to step {jump_back_to}")
        
        # Jump back by adding noise
        t_jump_back = torch.tensor([jump_back_to], device=device)
        
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
    return x_t, current_step


def repaint_inpaint(
        image_path: str,
        mask_path: str,
        example_name: str,
        example_output_dir: str,
        images_dir: str,
        diffusion_steps: int,
        model,
        diffusion,
        device,
        jump_length: int = 10,
        jump_n_sample: int = 10,
        debugging: bool = False,
        save_intermediate: bool = False,
    ):
    """
    RePaint algorithm implementation.
    
    The RePaint paper introduces a projection logic and jumping mechanism where every jump_length iterations, the scheduler jumps back by jump_n_sample steps and then continues forward. These help blending the inpainted and known regions.
    """
    if debugging:
        logger.log(f"---[RePaint] image: {image_path}---")


    t0 = time.time()
    x0 = load_image(image_path).to(device)
    t1 = time.time()
    H, W = x0.shape[-2], x0.shape[-1]

    output_dir = example_output_dir or images_dir
    os.makedirs(output_dir, exist_ok=True)

    if debugging:
        logger.log(f"[Timing] Loading image took {t1 - t0:.2f} seconds.")

    # Visualize input image
    t2 = time.time()
    if debugging:
        save_image(x0, os.path.join(output_dir, "input.png"))
        logger.log(f"Input image saved.")
    t3 = time.time()
    if debugging:
        logger.log(f"[Timing] Saving input image took {t3 - t2:.2f} seconds.")

    try:
        t4 = time.time()
        mask = load_mask(mask_path, (H, W), device=device)
        t5 = time.time()
        if debugging:
            logger.log(f"[Timing] Loading mask took {t5 - t4:.2f} seconds.")
    except Exception as e:
        logger.log(f"Error with --mask_path: {mask_path}")
        return

    # Visualize mask
    t6 = time.time()
    if debugging:
        save_image(mask, os.path.join(output_dir, "mask.png"))
        logger.log(f"Mask image saved.")
    t7 = time.time()
    if debugging:
        logger.log(f"[Timing] Saving mask image took {t7 - t6:.2f} seconds.")

    # Visualize input with bounding boxes
    t8 = time.time()
    if debugging:
        draw_bbox(image_path, example_name, os.path.join(output_dir, "input_with_bbox.png"))
        logger.log(f"Input image with bounding boxes saved.")
    t9 = time.time()
    if debugging:
        logger.log(f"[Timing] Drawing bbox took {t9 - t8:.2f} seconds.")

    # Start from noise, project known region at T
    with torch.no_grad():
        t10 = time.time()
        T = diffusion_steps - 1
        t = torch.tensor([T], device=device)
        x_t = torch.randn_like(x0)
        known_noise = torch.randn_like(x0)
        x_t = project_known_region(diffusion, x0, x_t, t, mask, noise=known_noise)
        t11 = time.time()
        if debugging:
            logger.log(f"[Timing] Initial noise and projection took {t11 - t10:.2f} seconds.")

        if debugging:
            if jump_length > 0:
                logger.log(f"[RePaint] reverse diffusion with {diffusion_steps} steps, jumping every {jump_length} steps.")
            else:
                logger.log(f"[RePaint] reverse diffusion with {diffusion_steps} steps and no jumping.")

        current_step = diffusion_steps - 1

        # tqdm progress bar for the diffusion steps
        diffusion_start = time.time()
        for _ in trange(diffusion_steps, desc="RePaint Diffusion", leave=False):
            step_start = time.time()
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

            # Move to next step
            current_step -= 1

            # RePaint jumping mechanism: every jump_length iterations, jump back
            if jump_length > 0 and (diffusion_steps - (current_step + 1)) % jump_length == 0 and current_step >= 0:
                x_t, current_step = repaint_jump_step(
                    x_t, current_step, jump_n_sample, diffusion_steps,
                    diffusion, device, debugging, logger, x0, mask, known_noise
                )

            # Save intermediate results
            if save_intermediate and (diffusion_steps - (current_step + 1)) % 100 == 0:
                save_image(x_t, os.path.join(output_dir, f"repaint_t{current_step:04d}.png"))
                if debugging:
                    logger.log(f"[RePaint] saved intermediate at step {current_step}")
            step_end = time.time()
            if debugging:
                logger.log(f"[Timing] Diffusion step took {step_end - step_start:.2f} seconds.")

        diffusion_end = time.time()
        if debugging:
            logger.log(f"[Timing] All diffusion steps took {diffusion_end - diffusion_start:.2f} seconds.")

        # Project known region one last time
        t12 = time.time()
        final_img = x_t * (1 - mask) + x0 * mask
        t13 = time.time()
        if debugging:
            logger.log(f"[Timing] Final projection took {t13 - t12:.2f} seconds.")

        if debugging:
            final_path = os.path.join(output_dir, "repaint_result.png")
            save_image(final_img, final_path)
            logger.log(f"[RePaint] saved: {final_path}")

        # Save final image
        final_general_path = os.path.join(CF_DIR, example_name)
        save_image(final_img, final_general_path)

        # Visualize output with bounding boxes
        if debugging:
            draw_bbox(final_path, example_name, os.path.join(output_dir, "output_with_bbox.png"))


# ---------------------------
# CLI
# ---------------------------
def main():
    # Parse args and setup logging

    t0 = time.time()
    args = create_argparser().parse_args()
    t1 = time.time()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")
    
    output_dir = os.path.join(LOGS_PATH, f"RePaint_logs/job_{slurm_job_id}_{timestamp}")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    t2 = time.time()
    device = dist_util.dev()
    dist_util.setup_dist()
    logger.configure(experiment_type="repaint")
    t3 = time.time()

    if getattr(args, 'debugging', False):
        logger.log(f"[Timing] Argparse took {t1 - t0:.2f} seconds.")
        logger.log(f"[Timing] Output dir setup took {t2 - t1:.2f} seconds.")
        logger.log(f"[Timing] Device/dist_util/logger setup took {t3 - t2:.2f} seconds.")

    t4 = time.time()
    model, diffusion = load_model(args, device)
    t5 = time.time()
    os.makedirs(CF_DIR, exist_ok=True)

    if getattr(args, 'debugging', False):
        logger.log(f"[Timing] Model and diffusion loading took {t5 - t4:.2f} seconds.")

    t6 = time.time()
    for i, example_name in enumerate(args.examples):
        if example_name in os.listdir(CF_DIR):
            if args.debugging:
                logger.log(f"[RePaint] counterfactual already created for: {example_name}")
            continue
        
        image_path = os.path.join(DATASET_DIR, example_name)
        mask_path = os.path.join(MASKS_DIR, example_name)
        example_name = example_name
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        
        example_output_dir = os.path.join(
            images_dir, f"example_{i+1:02d}_{os.path.splitext(example_name)[0]}"
        )

        repaint_inpaint(image_path, mask_path, example_name, example_output_dir, images_dir, args.diffusion_steps, model,
        diffusion, device, args.jump_length, args.jump_n_sample, args.debugging,
        args.save_intermediate)
        break  # Only process one image then stop
    t7 = time.time()
    if getattr(args, 'debugging', False):
        logger.log(f"[Timing] All examples processed in {t7 - t6:.2f} seconds.")


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
        use_fp16=False,
        jump_length=-1,
        jump_n_sample=-1,
        debugging=True,
        save_intermediate=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()