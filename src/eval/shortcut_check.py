"""Sanity check: is the model learning real AI artifacts, or a dataset
shortcut (e.g. compression level correlating with label)? Per the workshop's
DDA insight (NeurIPS 2025) warning about frequency/compression bias.

Checks:
  1. Does predicted probability correlate with JPEG quality of the INPUT
     image, independent of the true label? (estimate via file-size proxy or
     re-compression detection)
  2. Does accuracy differ suspiciously between data sources (e.g. near-perfect
     on one dataset, poor on another) in a way that suggests the model latched
     onto source-specific artifacts rather than general AI signal?

This is a diagnostic tool, not a formal statistical test — read the printed
correlations and per-source breakdown critically rather than trusting a
single threshold.

Usage:
    python src/eval/shortcut_check.py --checkpoint outputs/baseline/model_best.pt
"""
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pointbiserialr

from src.data.dataset import split_samples
from src.eval.data_compat import scan_all_sources, load_eval_model, EvalDataset
from src.utils import load_config, get_logger

logger = get_logger(__name__)


def estimate_jpeg_quality_proxy(path: str) -> float:
    """Cheap proxy for compression level: file size relative to pixel count.
    Lower value ~ more compressed. Not a precise quality estimate, but enough
    to check for gross correlation with predictions.

    Only meaningful for plain on-disk files (CIFAKE) — SID-Set/WildFake
    samples carry a pseudo-path (see src/eval/data_compat.py) with no real
    file to stat, so this returns NaN for those and they're excluded from
    this particular check below (the per-source breakdown, Check 2, has no
    such limitation and still covers every source).
    """
    import os
    from PIL import Image

    if "://" in path:
        return float("nan")
    try:
        size_bytes = os.path.getsize(path)
        with Image.open(path) as img:
            pixels = img.size[0] * img.size[1]
        return size_bytes / max(1, pixels)
    except Exception:
        return float("nan")


def run_shortcut_check(config: dict, checkpoint_path: str, device: str = "cpu"):
    model = load_eval_model(checkpoint_path, config, device=device)
    samples = scan_all_sources(config)
    splits = split_samples(
        samples,
        holdout_generator=config["data"]["holdout_generator"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        seed=config["seed"],
    )
    test_ds = EvalDataset(splits["test"], config, eval_transform_name="clean")
    loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    probs, labels, paths, generators = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float().to(device)
            p = model.predict_proba(images).cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(batch["label"].numpy().tolist())
            paths.extend(batch["path"])
            generators.extend(batch["generator"])

    # --- Check 1: compression-proxy correlation, controlling loosely for label ---
    compression_proxy = [estimate_jpeg_quality_proxy(p) for p in paths]
    valid = [i for i, c in enumerate(compression_proxy) if not np.isnan(c)]
    if len(valid) > 10:
        corr, pval = pointbiserialr(
            [labels[i] for i in valid], [compression_proxy[i] for i in valid]
        )
        logger.info(
            "Compression-proxy vs. TRUE LABEL correlation: r=%.3f (p=%.4f). "
            "If |r| is large, real/AI classes may differ systematically in "
            "compression, letting the model shortcut on that instead of real artifacts.",
            corr, pval,
        )
        corr_pred, pval_pred = pointbiserialr(
            [1 if probs[i] >= 0.5 else 0 for i in valid], [compression_proxy[i] for i in valid]
        )
        logger.info(
            "Compression-proxy vs. MODEL PREDICTION correlation: r=%.3f (p=%.4f).",
            corr_pred, pval_pred,
        )

    # --- Check 2: per-generator/source accuracy breakdown ---
    by_source = defaultdict(list)
    for gen, label, prob in zip(generators, labels, probs):
        pred = 1 if prob >= 0.5 else 0
        by_source[gen].append(int(pred == label))

    logger.info("Per-source accuracy breakdown (large gaps may indicate source-specific shortcuts):")
    for source, correct in sorted(by_source.items()):
        logger.info("  %-25s acc=%.3f (n=%d)", source, np.mean(correct), len(correct))


def main():
    parser = argparse.ArgumentParser(description="Run shortcut-learning sanity checks.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    run_shortcut_check(config, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
