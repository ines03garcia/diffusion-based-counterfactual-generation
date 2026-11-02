
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(
        description="Upload a trained DDPM checkpoint to Hugging Face"
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository (username/model-name)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/model008000.pt",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to training args.txt",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )

    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    api = HfApi()

    # Create repository if it doesn't exist
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # Upload model checkpoint
    print(f"Uploading {checkpoint}...")

    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=checkpoint.name,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload trained DDPM checkpoint",
    )

    # Upload training configuration if available
    if args.config:
        config_path = Path(args.config)

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Config not found: {config_path}"
            )

        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="training_args.json",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload DDPM training configuration",
        )

    # Create README only if it doesn't exist
    existing_files = api.list_repo_files(args.repo_id)

    if "README.md" not in existing_files:
        readme = """---
tags:
- diffusion
- ddpm
- pytorch
---

# DDPM - Healthy Images

Denoising Diffusion Probabilistic Model trained on healthy images.

## Architecture

Custom DDPM based on OpenAI's guided_diffusion implementation.

## Checkpoint

`model008000.pt`

## Loading

Requires the original guided_diffusion implementation
and the training configuration used to create the model.
"""

        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add DDPM model card",
        )

    print("Upload complete!")
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
