"""
Noise and denoise some images using a trained DDPM model.
"""
import torch
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from src.DDPM.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from src.DDPM.guided_diffusion import dist_util, logger
from src.config import DATASET_DIR, MODELS_ROOT


def load_image(path):
    """Load and preprocess grayscale image to match training normalization"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    img = Image.open(path).convert("L")
    img = np.array(img).astype(np.float32) / 255.0  # [0, 255] -> [0, 1]
    if img.ndim != 2:
        raise ValueError(f"Expected grayscale image with 2 dimensions, got shape {img.shape}")
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    return img

def save_image(tensor, path):
    """Save tensor as grayscale image"""
    path = Path(path)
    img = tensor.squeeze().cpu().numpy()  # Move to CPU first
    if img.ndim != 2:
        raise ValueError(f"Expected tensor to be 2D after squeeze, got shape {img.shape}")
    img = np.clip(img * 255, 0, 255).astype(np.uint8)  # [0, 1] -> [0, 255]
    Image.fromarray(img, mode='L').save(path)

def load_model(args, device):
    """Load trained model from checkpoint"""
    if args.debugging:
        logger.log("creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Move model to device
    model.to(device)

    # Load checkpoint if provided
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if args.debugging:
            logger.log(f"loading model from checkpoint: {args.model_path}")
        
        state_dict = dist_util.load_state_dict(str(model_path), map_location=device)
        model.load_state_dict(state_dict)

        if args.debugging:
            logger.log("model loaded successfully")

    model.eval()
    return model, diffusion


def main():
    args = create_argparser().parse_args()

    # Setup device
    device = dist_util.dev()
    if args.debugging:
        logger.log("Using device:", device)
    dist_util.setup_dist()

    # Setup logging
    args.output_dir = Path(logger.configure(experiment_type="ddpm_noise_and_denoise"))
    args.images_dir = args.output_dir / "images"

    logger.log(f"\nOutput directory: {args.output_dir}")
    logger.log(f"\nProcessing {len(args.examples)} example images...")

    # Load model and diffusion
    model, diffusion = load_model(args, device)

    for i, example_name in enumerate(tqdm(args.examples, desc="Images", unit="img")):
        if args.debugging:
            logger.log(f"\n--- Processing example {i+1}/{len(args.examples)}: {example_name} ---")

        image_path = Path(DATASET_DIR) / example_name

        if not image_path.exists():
            logger.log(f"Warning: Image not found: {image_path}")
            continue

        if args.debugging:
            example_name_no_ext = Path(example_name).stem
            output_dir = args.images_dir / f"example_{i+1:02d}_{example_name_no_ext}"
        else:
            output_dir = args.images_dir

        process_single_image(
            example_name=example_name,
            image_path=image_path,
            output_dir=output_dir,
            args=args,
            model=model,
            diffusion=diffusion,
            device=device
        )
        if args.debugging:
            logger.log(f"Successfully processed: {example_name}")

    if args.debugging:
        logger.log("\nFinished processing all examples")



def process_single_image(example_name, image_path, output_dir, args, model, diffusion, device):
    """Process a single input image through noise and denoise"""
    x_start = load_image(str(image_path)).to(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original input 
    if args.debugging:
        input_path = output_dir / "input.png"
        save_image(x_start, str(input_path))
        logger.log(f"Saved input image")

    # Add noise
    t_noise = torch.tensor([args.noise_steps - 1], device=device)  # 0-based indexing
    noise = torch.randn_like(x_start)
    x_noisy = diffusion.q_sample(x_start=x_start, t=t_noise, noise=noise)

    # Save noisy image
    if args.debugging:
        noisy_path = output_dir / f"noisy_t{args.noise_steps}.png"
        save_image(x_noisy, str(noisy_path))
        logger.log(f"Saved noisy image")

        logger.log(f"Starting denoising for {args.noise_steps} steps...")

    # Denoise
    sample = x_noisy
    denoise_iter = tqdm(reversed(range(args.noise_steps)), desc="Denoising", unit="step")
    for t in denoise_iter:
        if args.debugging and (t % 50 == 0):
            denoise_iter.set_postfix_str(f"step {t}")

        t_tensor = torch.tensor([t], device=device)
        with torch.no_grad():
            out = diffusion.p_sample(model, sample, t_tensor)
            sample = out["sample"]

        # Save intermediate images every 50 steps if debugging
        if args.debugging and (t % 50 == 0 or t == 0):
            interm_path = output_dir / f"intermediate_t{t}.png"
            save_image(sample, str(interm_path))
            logger.log(f"Saved intermediate image at t={t}")

    # Save denoised image
    denoised_path = output_dir / f"denoised_from_t{args.noise_steps}.png"
    save_image(sample, str(denoised_path))

    if args.debugging:
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
        model_path=str(Path(MODELS_ROOT) / "model009000.pt"),
        examples = [
            "008c66563c73b2f5b8e42915b2cd6af5.png", "00be38a5c0566291168fe381ba0028e6.png", "00ec2be128f964da6f0b0ba179c4d138.png",
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
        image_path=DATASET_DIR,
        noise_steps=200,  # How many noise steps to add
        
        # Device settings
        use_fp16=False,
        debugging=True, # Enable extra messages and images
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()