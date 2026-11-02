"""Upload image datasets to a Hugging Face dataset repository."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import HfApi

from src.config import (
    CF_DIR,
    DATASET_DIR,
    ROOT,
    UPSCALED_CF_DIR,
    UPSCALED_DATASET_DIR,
)


DEFAULT_DATASETS = {
    "vindr_mammo_clip_512_clahe": DATASET_DIR,
    "vindr_mammo_clip_512_clahe_912_1520": UPSCALED_DATASET_DIR,
    "repaint_results": CF_DIR,
    "repaint_results_912x1520": UPSCALED_CF_DIR,
}

IMAGE_PATTERNS = [
    "**/*.png",
]


def parse_dataset_spec(spec: str) -> tuple[str, Path]:
    """Parse a NAME=PATH dataset override."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Dataset must use NAME=PATH format")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Dataset NAME and PATH cannot be empty")
    return name, Path(path)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(ROOT) / path


def image_count(folder: Path) -> int:
    return sum(1 for path in folder.rglob("*.png") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload image folders to a Hugging Face dataset repository."
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Hugging Face dataset repository, for example username/mammogram-datasets.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_spec,
        metavar="NAME=PATH",
        help=(
            "Dataset override. Repeat this option to upload selected folders. "
            "When omitted, the four configured image folders are uploaded."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repository as private if it does not already exist.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face branch or revision to upload to.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate folders and report image counts without uploading.",
    )
    args = parser.parse_args()

    datasets = dict(args.dataset) if args.dataset else {
        name: Path(path) for name, path in DEFAULT_DATASETS.items()
    }
    resolved_datasets = {
        name: resolve_path(path) for name, path in datasets.items()
    }

    missing = [path for path in resolved_datasets.values() if not path.is_dir()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Dataset folder(s) not found:\n{missing_text}")

    counts = {name: image_count(path) for name, path in resolved_datasets.items()}
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"Dataset folder(s) contain no supported image files: {empty}")

    for name, path in resolved_datasets.items():
        print(f"{name}: {counts[name]} image(s) in {path}")

    if args.dry_run:
        print("Dry run complete; nothing was uploaded.")
        return

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    for name, folder in resolved_datasets.items():
        print(f"Uploading {folder} to {args.repo_id}:{name}/...")
        upload_kwargs = {
            "repo_id": args.repo_id,
            "repo_type": "dataset",
            "folder_path": str(folder),
            "path_in_repo": name,
            "allow_patterns": IMAGE_PATTERNS,
            "commit_message": f"Upload image dataset: {name}",
        }
        if args.revision is not None:
            upload_kwargs["revision"] = args.revision
        api.upload_folder(**upload_kwargs)

    print(f"Uploaded {len(resolved_datasets)} image dataset(s).")
    print(f"https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
