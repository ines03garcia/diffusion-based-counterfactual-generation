"""
Train a diffusion model on healthy images.
"""
import torch.nn as nn
import argparse
import json
import os

from code.DDPM.guided_diffusion.image_datasets import load_data
from code.DDPM.guided_diffusion import dist_util, logger
from code.DDPM.guided_diffusion.resample import create_named_schedule_sampler
from code.DDPM.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from code.DDPM.guided_diffusion.train_util import TrainLoop
from code.config import DATASET_DIR



def main():
    args = create_argparser().parse_args()
    
    # Add job ID to experiment name if running in SLURM
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    args.experiment_name = f"{args.experiment_name}_job{job_id}"
    
    print(vars(args))

    dist_util.setup_dist()
    logger.configure(experiment_name=args.experiment_name)

    # Create path logs/<experiment_name>/args.txt and writes the args
    args_path = os.path.join(logger.get_dir(), 'args.txt') 
    with open(args_path, 'w') as convert_file:
            convert_file.write(json.dumps(vars(args)))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    if args.gpus > 1:
        model = nn.DataParallel(model)
        print("Using cuda!")
        model = model.to("cuda")
    else:
        device = dist_util.dev()
        print("Using device:", device)
        model = model.to(device)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        category="healthy"
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        image_size=args.image_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        gpus=1,
        data_dir=DATASET_DIR,
        experiment_name='ddpm_train',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.05,
        lr_anneal_steps=1000000,
        batch_size=16,
        microbatch=4,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()