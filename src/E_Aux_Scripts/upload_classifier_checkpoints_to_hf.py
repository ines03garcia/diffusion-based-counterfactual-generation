"""Upload the identified classifier checkpoints to a Hugging Face model repo."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import HfApi

from src.config import ROOT


DEFAULT_CHECKPOINTS = {
    "convnext_baseline": (
        "data/logs/Classifiers/train/birads/convnext/baseline_aug/"
        "single_seed_no_validation/local_24-06-2026_16:21:32/seed_0/final_model.pth"
    ),
    "convnext_cf": (
        "data/logs/Classifiers/train/birads/convnext/cf_aug/"
        "single_seed_no_validation/local_24-06-2026_16:18:49/seed_0/final_model.pth"
    ),
    "vit_baseline": (
        "data/logs/Classifiers/train/birads/vit/baseline_aug/"
        "single_seed_no_validation/local_19-06-2026_10:45:32/seed_0/final_model.pth"
    ),
    "vit_cf": (
        "data/logs/Classifiers/train/birads/vit/cf_aug/"
        "single_seed_no_validation/local_20-06-2026_20:17:55/seed_0/final_model.pth"
    ),
    "mammo_clip_baseline": (
        "data/logs/Classifiers/train/birads/mammo-clip/baseline_aug/"
        "single_seed_no_validation/local_29-06-2026_14:12:41/seed_0/checkpoint_epoch_30.pth"
    ),
    "mammo_clip_cf": (
        "data/logs/Classifiers/train/birads/mammo-clip/cf_aug/"
        "single_seed_no_validation/local_29-06-2026_14:43:07/seed_0/checkpoint_epoch_30.pth"
    ),
}

DEFAULT_ARGS = {
    name: path.split("/seed_0/", 1)[0] + "/args.json"
    for name, path in DEFAULT_CHECKPOINTS.items()
}


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    """Parse a NAME=PATH checkpoint override."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Checkpoint must use NAME=PATH format")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Checkpoint NAME and PATH cannot be empty")
    return name, Path(path)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(ROOT) / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload classifier checkpoints to a Hugging Face model repository."
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Hugging Face repository, for example username/classifier-checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_spec,
        metavar="NAME=PATH",
        help=(
            "Checkpoint override. Repeat this option to upload selected files. "
            "When omitted, the six matched baseline/CF checkpoints are uploaded."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private if it does not already exist.",
    )
    args = parser.parse_args()

    checkpoints = dict(args.checkpoint) if args.checkpoint else {
        name: Path(path) for name, path in DEFAULT_CHECKPOINTS.items()
    }
    resolved_checkpoints = {
        name: resolve_path(path) for name, path in checkpoints.items()
    }

    missing = [path for path in resolved_checkpoints.values() if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Checkpoint(s) not found:\n{missing_text}")

    args_paths = {
        name: resolve_path(Path(DEFAULT_ARGS[name]))
        for name in resolved_checkpoints
        if name in DEFAULT_ARGS
    }
    missing_args = [path for path in args_paths.values() if not path.is_file()]
    if missing_args:
        missing_text = "\n".join(str(path) for path in missing_args)
        raise FileNotFoundError(f"Training args file(s) not found:\n{missing_text}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    for name, checkpoint in resolved_checkpoints.items():
        path_in_repo = f"checkpoints/{name}.pth"
        print(f"Uploading {checkpoint} as {path_in_repo}...")
        api.upload_file(
            path_or_fileobj=str(checkpoint),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload classifier checkpoint: {name}",
        )

        args_path = args_paths.get(name)
        if args_path is not None:
            config_path_in_repo = f"configs/{name}_args.json"
            print(f"Uploading {args_path} as {config_path_in_repo}...")
            api.upload_file(
                path_or_fileobj=str(args_path),
                path_in_repo=config_path_in_repo,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Upload training arguments: {name}",
            )

    print(f"Uploaded {len(resolved_checkpoints)} checkpoint(s).")
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
