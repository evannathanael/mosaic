"""Download the datasets used by this project.

Usage:
    python src/data/download.py --dataset all --out data/raw
    python src/data/download.py --dataset sid_set --out data/raw

NOTE: WildFake is hosted on ModelScope and its dataset card must be translated
via the translate button before use, per the challenge instructions — this
script prints a reminder rather than silently downloading un-reviewed content.
"""
import argparse
from pathlib import Path

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

DATASETS = {
    "sid_set": {
        "source": "huggingface",
        "repo_id": "saberzl/SID_Set",
        "note": "Image-level AIGC detection dataset.",
    },
    "cifake": {
        "source": "kaggle",
        "handle": "birdy654/cifake-real-and-ai-generated-synthetic-images",
        "note": "120K 32x32 real vs AI images (CIFAR-10 based).",
    },
    "wildfake": {
        "source": "modelscope",
        "handle": "hy2628982280/WildFake",
        "note": (
            "IMPORTANT: translate the dataset page via the translate button "
            "on ModelScope before use, per the challenge instructions."
        ),
    },
}


def download_sid_set(out_dir: Path):
    from huggingface_hub import snapshot_download

    logger.info("Downloading SID_Set from Hugging Face...")
    snapshot_download(
        repo_id=DATASETS["sid_set"]["repo_id"],
        repo_type="dataset",
        local_dir=str(out_dir / "sid_set"),
    )
    logger.info("SID_Set downloaded to %s", out_dir / "sid_set")


def download_cifake(out_dir: Path):
    import kagglehub

    logger.info("Downloading CIFAKE from Kaggle...")
    path = kagglehub.dataset_download(DATASETS["cifake"]["handle"])
    logger.info("CIFAKE downloaded to %s (copy/symlink into %s)", path, out_dir / "cifake")


def download_wildfake(out_dir: Path):
    logger.warning(
        "WildFake is hosted on ModelScope: https://modelscope.cn/datasets/%s/summary",
        DATASETS["wildfake"]["handle"],
    )
    logger.warning(
        "Please translate the dataset page via the translate button before "
        "downloading, per the challenge instructions, then place files under %s",
        out_dir / "wildfake",
    )


DOWNLOADERS = {
    "sid_set": download_sid_set,
    "cifake": download_cifake,
    "wildfake": download_wildfake,
}


def main():
    parser = argparse.ArgumentParser(description="Download project datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download.",
    )
    parser.add_argument("--out", default="data/raw", help="Output directory.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        logger.info("=== %s: %s ===", name, DATASETS[name]["note"])
        DOWNLOADERS[name](out_dir)

    logger.info(
        "Done. Remember: the validation subset (COCO val2017 non-AIGC + "
        "DALL-E Advanced AIGC) is for demonstration only — never train on it."
    )


if __name__ == "__main__":
    main()
