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
from tqdm import trange, tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import time

from src.DDPM.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from src.DDPM.guided_diffusion import dist_util, logger
from src.config import MODELS_ROOT, METADATA_ROOT, MASKS_DIR, DATASET_DIR


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
        image_row.loc[:, col] = image_row[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

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

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)

    if args.model_path:
        if args.debugging:
            logger.log(f"Loading model from checkpoint: {args.model_path}")
        state_dict = dist_util.load_state_dict(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
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
    batch_size = x_t.shape[0]
    
    if jump_back_to > current_step:
        if debugging:
            logger.log(f"[RePaint] Jumping back from step {current_step} to step {jump_back_to}")
        
        # Jump back by adding noise
        t_jump_back = torch.tensor([jump_back_to] * batch_size, device=device)
        
        # Calculate the number of steps to jump back
        steps_back = jump_back_to - current_step
        
        # Add noise to simulate going back in time
        jump_noise = torch.randn_like(x_t)
        
        # Use the forward process to add noise (q_sample)
        # We go from current denoised state back to a more noisy state
        x_t = diffusion.q_sample(x_start=x_t, t=torch.tensor([steps_back] * batch_size, device=device), noise=jump_noise)
        
        # Project the known region at the jumped-back timestep
        x_t = project_known_region(diffusion, x0, x_t, t_jump_back, mask, noise=known_noise)
        
        # Update current step to the jumped-back position
        current_step = jump_back_to
    return x_t, current_step


def _load_single_image_mask(idx, image_path, mask_path, example_name, device):
    """Helper function to load a single image and mask pair."""
    try:
        x0 = load_image(image_path)
        H, W = x0.shape[-2], x0.shape[-1]
        mask = load_mask(mask_path, (H, W), device="cpu")  # Load to CPU first
        return idx, x0, mask, None
    except Exception as e:
        return idx, None, None, str(e)


def repaint_inpaint_batch(
        image_paths: list,
        mask_paths: list,
        example_names: list,
        diffusion_steps: int,
        model,
        diffusion,
        device,
        repaint_results_dir: str,
        debugging_dir: str = None,
        jump_length: int = 10,
        jump_n_sample: int = 10,
        debugging: bool = False,
        save_intermediate: bool = False,
    ):
    """
    RePaint algorithm implementation for batch processing.
    
    The RePaint paper introduces a projection logic and jumping mechanism where every jump_length iterations, 
    the scheduler jumps back by jump_n_sample steps and then continues forward. These help blending the 
    inpainted and known regions.
    """
    batch_size = len(image_paths)
    
    if debugging:
        logger.log(f"---[RePaint] Processing batch of {batch_size} images---")

    # Load all images and masks in parallel
    x0_list = []
    mask_list = []
    valid_indices = []
    
    with ThreadPoolExecutor(max_workers=min(8, batch_size)) as executor:
        futures = {
            executor.submit(_load_single_image_mask, idx, img_path, mask_path, name, device): idx
            for idx, (img_path, mask_path, name) in enumerate(zip(image_paths, mask_paths, example_names))
        }
        
        results = []
        for future in as_completed(futures):
            results.append(future.result())
        
        # Sort by original index to maintain order
        results.sort(key=lambda x: x[0])
        
        for idx, x0, mask, error in results:
            if error is not None:
                logger.log(f"Error loading {image_paths[idx]} or {mask_paths[idx]}: {error}")
                continue
            
            # Keep on CPU for now
            x0_list.append(x0)
            mask_list.append(mask)
            valid_indices.append(idx)
            
            if debugging:
                example_name = example_names[idx]
                sample_output_dir = os.path.join(debugging_dir, example_name.split(".")[0])
                os.makedirs(sample_output_dir, exist_ok=True)

                save_image(x0, os.path.join(sample_output_dir, "input.png"))
                save_image(mask, os.path.join(sample_output_dir, "mask.png"))
                draw_bbox(image_paths[idx], example_name, os.path.join(sample_output_dir, "input_with_bbox.png"))
    
    if len(x0_list) == 0:
        logger.log("No valid images in batch, skipping.")
        return
    
    # Stack into batched tensors on CPU, then move to GPU
    x0_batch = torch.cat(x0_list, dim=0).to(device)  # (B, 1, H, W)
    mask_batch = torch.cat(mask_list, dim=0).to(device)  # (B, 1, H, W)
    
    # Start from noise, project known region at T
    with torch.no_grad():
        T = diffusion_steps - 1
        t = torch.tensor([T] * x0_batch.shape[0], device=device)
        x_t = torch.randn_like(x0_batch)
        known_noise = torch.randn_like(x0_batch)
        x_t = project_known_region(diffusion, x0_batch, x_t, t, mask_batch, noise=known_noise)

        if debugging:
            if jump_length > 0:
                logger.log(f"[RePaint] reverse diffusion with {diffusion_steps} steps, jumping every {jump_length} steps.")
            else:
                logger.log(f"[RePaint] reverse diffusion with {diffusion_steps} steps and no jumping.")

        current_step = diffusion_steps - 1

        # tqdm progress bar for the diffusion steps
        for _ in trange(diffusion_steps, desc=f"RePaint Batch ({len(x0_list)} imgs)", leave=False):
            t = torch.tensor([current_step] * x0_batch.shape[0], device=device)

            # Project known region at current t
            x_t = project_known_region(diffusion, x0_batch, x_t, t, mask_batch, noise=known_noise)

            # One denoising step
            out = diffusion.p_sample(model, x_t, t)
            x_tm1 = out["sample"]

            # Project at t-1 if not at the final step
            if current_step > 0:
                t_prev = torch.tensor([current_step - 1] * x0_batch.shape[0], device=device)
                x_tm1 = project_known_region(diffusion, x0_batch, x_tm1, t_prev, mask_batch, noise=known_noise)

            x_t = x_tm1

            # Move to next step
            current_step -= 1

            # RePaint jumping mechanism: every jump_length iterations, jump back
            if jump_length > 0 and (diffusion_steps - (current_step + 1)) % jump_length == 0 and current_step >= 0:
                x_t, current_step = repaint_jump_step(
                    x_t, current_step, jump_n_sample, diffusion_steps,
                    diffusion, device, debugging, logger, x0_batch, mask_batch, known_noise
                )

            # Save intermediate results (only for debugging and first image in batch)
            if save_intermediate and debugging and (diffusion_steps - (current_step + 1)) % 100 == 0:
                example_name = example_names[valid_indices[0]]
                sample_output_dir = os.path.join(debugging_dir, example_name.split(".")[0])
                save_image(x_t[0:1], os.path.join(sample_output_dir, f"repaint_t{current_step:04d}.png"))

        # Project known region one last time
        final_batch = x_t * (1 - mask_batch) + x0_batch * mask_batch

        # Save all results
        for idx, valid_idx in enumerate(valid_indices):
            example_name = example_names[valid_idx]
            final_img = final_batch[idx:idx+1]
            
            if debugging:
                sample_output_dir = os.path.join(debugging_dir, example_name.split(".")[0])
                final_path = os.path.join(sample_output_dir, "repaint_result.png")
                save_image(final_img, final_path)
                draw_bbox(final_path, example_name, os.path.join(sample_output_dir, "output_with_bbox.png"))
                logger.log(f"[RePaint] saved: {final_path}")

            # Save final image to repaint results directory
            final_cfs_path = os.path.join(repaint_results_dir, example_name)
            save_image(final_img, final_cfs_path)


# ---------------------------
# CLI
# ---------------------------
def main():
    # Parse args
    args = create_argparser().parse_args()

    device = dist_util.dev()
    dist_util.setup_dist()

    output_dir = logger.configure(experiment_type="repaint")
    repaint_results_dir = os.path.join(output_dir, "images/repaint_results")
    os.makedirs(repaint_results_dir, exist_ok=True)

    if args.debugging:
        debugging_dir = os.path.join(output_dir, "images/debugging")
        os.makedirs(debugging_dir, exist_ok=True)
        logger.log(f"Debugging mode enabled. Debugging outputs will be saved to {debugging_dir}")

    model, diffusion = load_model(args, device)
    
    logger.log(f"[INFO] Diffusion model loaded with {diffusion.num_timesteps} timesteps (timestep_respacing='{args.timestep_respacing}')")

    # Filter out already processed images if resume_repaint is enabled
    if args.resume_repaint:
        existing_cfs = set(os.listdir(args.cf_dir))
        examples_to_process = [name for name in args.examples if name not in existing_cfs]
        
        if args.debugging:
            logger.log(f"Resume mode: Found {len(existing_cfs)} existing counterfactuals in {args.cf_dir}")
            logger.log(f"Processing {len(examples_to_process)} new images")
    else:
        examples_to_process = args.examples
        if args.debugging:
            logger.log(f"Processing all {len(examples_to_process)} images")

    start_time = time.time()
    batch_size = args.batch_size
    
    # Process in batches
    for batch_start in tqdm(range(0, len(examples_to_process), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(examples_to_process))
        batch_names = examples_to_process[batch_start:batch_end]
        
        batch_image_paths = [os.path.join(DATASET_DIR, name) for name in batch_names]
        batch_mask_paths = [os.path.join(MASKS_DIR, name) for name in batch_names]
        
        # Check if all files exist
        valid_batch = True
        for img_path, mask_path in zip(batch_image_paths, batch_mask_paths):
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                logger.log(f"File not found: {img_path} or {mask_path}")
                valid_batch = False
                break
        
        if not valid_batch:
            continue
        
        repaint_inpaint_batch(
            batch_image_paths, 
            batch_mask_paths, 
            batch_names, 
            diffusion.num_timesteps,  # Use actual timesteps from diffusion object
            model, 
            diffusion, 
            device, 
            repaint_results_dir,
            debugging_dir if args.debugging else None,
            args.jump_length, 
            args.jump_n_sample, 
            args.debugging, 
            args.save_intermediate
        )
        
        # Clear GPU cache between batches to prevent memory fragmentation
        torch.cuda.empty_cache()
    
    end_time = time.time()
    logger.log(f"Total RePaint time for {len(examples_to_process)} images: {end_time - start_time:.2f} seconds.")


def create_argparser():
    metadata = os.path.join(METADATA_ROOT, "resized_annotations_512.csv")
    df = pd.read_csv(metadata)
    anomalous_images = df['image_id'].tolist()

    # Start with model defaults, then override
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
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
        model_path=os.path.join(MODELS_ROOT, "model008000.pt"),
        examples=anomalous_images,
        use_fp16=False,
        jump_length=-1,
        jump_n_sample=-1,
        debugging=False,
        save_intermediate=False,
        batch_size=4,
        resume_repaint=False,
        cf_dir=None,
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()