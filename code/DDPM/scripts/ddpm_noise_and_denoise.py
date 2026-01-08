"""
Noise and denoise images using a trained DDPM model.
"""
import torch
from PIL import Image
import numpy as np
import argparse
import os
from code.DDPM.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from code.DDPM.guided_diffusion import dist_util, logger
from code.config import DATASET_DIR, MODELS_ROOT


def load_image(path):
    """Load and preprocess grayscale image to match training normalization"""
    img = Image.open(path).convert("L")
    img = np.array(img).astype(np.float32) / 255.0  # [0, 255] -> [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    
    # Match training normalization: divide by max value
    if img.max() > 0:
        img = img / img.max()  # This matches your training data loading
    
    return img

def save_image(tensor, path):
    """Save tensor as grayscale image"""
    img = tensor.squeeze().cpu().numpy()  # Move to CPU first
    img = np.clip(img * 255, 0, 255).astype(np.uint8)  # [0, 1] -> [0, 255]
    Image.fromarray(img, mode='L').save(path)

def load_model(args, device):
    """Load trained model from checkpoint"""
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Move model to device
    model.to(device)
    
    # Load checkpoint if provided
    if args.model_path:
        logger.log(f"loading model from checkpoint: {args.model_path}")
        state_dict = dist_util.load_state_dict(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.log("model loaded successfully")
    
    model.eval()
    return model, diffusion

def main():
    args = create_argparser().parse_args()

    # Get Slurm job ID and create output directory to match log structure
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    args.output_dir = f"DDPM_logs/job_{slurm_job_id}/noise_denoise_output"
    
    # Setup device
    device = dist_util.dev()
    print(f"Using device: {device}")
    
    # Setup logging
    dist_util.setup_dist()
    logger.configure()
    
    # Load model and diffusion
    model, diffusion = load_model(args, device)
    
    logger.log(f"Processing {len(args.examples)} example images...")
    for i, example_name in enumerate(args.examples):
        logger.log(f"Processing example {i+1}/{len(args.examples)}: {example_name}")
        
        # Set the image path for this example
        args.image_path = os.path.join(DATASET_DIR, example_name)
        
        # Check if file exists
        if not os.path.exists(args.image_path):
            logger.log(f"Warning: Image not found: {args.image_path}")
            continue
        
        # Create example-specific output directory
        example_name_no_ext = os.path.splitext(example_name)[0]
        args.example_output_dir = os.path.join(args.output_dir, f"example_{i+1:02d}_{example_name_no_ext}")
        
        try:
            process_single_image(args, model, diffusion, device)
            logger.log(f"Successfully processed: {example_name}")
        except Exception as e:
            logger.log(f"Error processing {example_name}: {str(e)}")
            continue
    
    logger.log("Finished processing all examples")


def process_single_image(args, model, diffusion, device):
    """Process a single input image through noise and denoise"""
    logger.log(f"Processing image: {args.image_path}")
    
    # Load image and move to device
    x_start = load_image(args.image_path).to(device)
    print(f"Image loaded with shape: {x_start.shape}")
    
    # Use example-specific output directory if available, otherwise use general output dir
    output_dir = getattr(args, 'example_output_dir', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save original input image
    input_path = os.path.join(output_dir, "input.png")
    save_image(x_start, input_path)
    logger.log(f"Saved input image: {input_path}")
    
    # Add noise
    t_noise = torch.tensor([args.noise_steps - 1], device=device)  # 0-based indexing
    noise = torch.randn_like(x_start)
    x_noisy = diffusion.q_sample(x_start=x_start, t=t_noise, noise=noise)
    
    # Save noisy image
    noisy_path = os.path.join(output_dir, f"noisy_t{args.noise_steps}.png")
    save_image(x_noisy, noisy_path)
    logger.log(f"Saved noisy image: {noisy_path}")
    
    # Denoise
    logger.log(f"Starting denoising for {args.noise_steps} steps...")
    sample = x_noisy
    for t in reversed(range(args.noise_steps)):
        if t % 50 == 0:  # More frequent progress updates
            print(f"Denoising step: {t}")
        t_tensor = torch.tensor([t], device=device)
        with torch.no_grad():
            out = diffusion.p_sample(model, sample, t_tensor)
            sample = out["sample"]
    
    # Save denoised image
    denoised_path = os.path.join(output_dir, f"denoised_from_t{args.noise_steps}.png")
    save_image(sample, denoised_path)
    logger.log(f"Saved denoised image: {denoised_path}")


def create_argparser():
    defaults = dict(
        # Model parameters
        image_size=256,
        in_channels=1,
        out_channels=1,
        num_channels=128,
        num_res_blocks=2,
        attention_resolutions="32,16,8",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_new_attention_order=False,
        use_checkpoint=True,
        channel_mult="",
        
        # Diffusion parameters
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        learn_sigma=False,
        use_ddim=False,
        clip_denoised=True,
        
        # Class conditioning
        class_cond=False,

        # Inference parameters
        model_path=os.path.join(MODELS_ROOT, "model070000.pt"),
        examples = ["008c66563c73b2f5b8e42915b2cd6af5.png", "00be38a5c0566291168fe381ba0028e6.png", "00ec2be128f964da6f0b0ba179c4d138.png",
        "001ade2a3cb53fd808bd2856a0df5413.png", 
        "01df962b078e38500bf9dd9969a50083.png",
        "008bc6050f6d31fc255e5d87bcc87ba2.png",
        "013e4b7bcdaf536c4e37b4125ab8148b.png",
        "019b9f6365fa641db040b5b643fadc42.png",
        "0171ab32059f4c226164a13c311f6824.png",
        "005869cc3078b5868e83197922b74c62.png",
        "00857417d07096982013956033da1f75.png",
        "01958718afdf303581e758cdf34eaf8a.png",
        "002074412a8fc178c271fb93b55c3e29.png",
        "005918369ec07b1aed37d1dd78bc57fe.png",
        "01599597388f3185563decc34945f6b3.png",
        "002460132586dc0c7b88a59dce6e77bd.png"
        ],  
        image_path=os.path.join(DATASET_DIR, ".png"),
        noise_steps=200,  # How many noise steps to add
        
        # Device settings
        use_fp16=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()