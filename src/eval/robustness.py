"""Builds the robustness evaluation table required by the challenge:
accuracy/AUC on clean images vs. each transform vs. unseen-generator.

Follows the workshop's exact scoring recipe:
    Final Score = 0.50 * AUC_clean + 0.50 * AUC_robust
(AUC_robust = mean AUC across all non-clean, non-unseen-generator conditions)

Usage:
    python src/eval/robustness.py --checkpoint outputs/baseline/model_best.pt --out outputs/robustness_table.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data.dataset import split_samples
from src.eval.data_compat import scan_all_sources, load_eval_model, EvalDataset
from src.utils import load_config, get_logger, ensure_dir

logger = get_logger(__name__)

# Named conditions -> (transform name used by named_eval_transform, dataset split to use)
CONDITIONS = [
    ("clean", "clean", "test"),
    ("jpeg_q30", "jpeg_q30", "test"),
    ("blur_sigma2", "blur_sigma2", "test"),
    ("resize_0.25x", "resize_0.25x", "test"),
    ("noise_0.10", "noise_0.10", "test"),
    ("color_jitter", "color_jitter", "test"),
    ("crop_80", "crop_80", "test"),
    ("unseen_generator", "clean", "unseen_generator"),
]


def evaluate_condition(model, samples, transform_name, config, device) -> dict:
    ds = EvalDataset(samples, config, eval_transform_name=transform_name)
    loader = DataLoader(
        ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=4, pin_memory=(device == "cuda"),
    )

    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float().to(device)
            probs = model.predict_proba(images).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch["label"].numpy().tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    return {
        "accuracy": accuracy_score(all_labels, preds) if all_labels else float("nan"),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan"),
        "n": len(all_labels),
    }


def build_robustness_table(config: dict, checkpoint_path: str, device: str = "cpu") -> pd.DataFrame:
    model = load_eval_model(checkpoint_path, config, device=device)

    samples = scan_all_sources(config)
    splits = split_samples(
        samples,
        holdout_generator=config["data"]["holdout_generator"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        seed=config["seed"],
    )

    rows = []
    for condition_name, transform_name, split_name in CONDITIONS:
        eval_samples = splits[split_name]
        if not eval_samples:
            logger.warning("Skipping '%s' — no samples in split '%s'", condition_name, split_name)
            continue
        metrics = evaluate_condition(model, eval_samples, transform_name, config, device)
        logger.info(
            "%-18s acc=%.3f  auc=%.3f  (n=%d)",
            condition_name, metrics["accuracy"], metrics["auc"], metrics["n"],
        )
        rows.append({"condition": condition_name, "accuracy": metrics["accuracy"], "auc": metrics["auc"], "n": metrics["n"]})

    df = pd.DataFrame(rows)

    # Final Score per the workshop formula
    clean_row = df[df["condition"] == "clean"]
    robust_rows = df[~df["condition"].isin(["clean", "unseen_generator"])]
    if not clean_row.empty and not robust_rows.empty:
        auc_clean = clean_row["auc"].iloc[0]
        auc_robust = robust_rows["auc"].mean()
        final_score = 0.5 * auc_clean + 0.5 * auc_robust
        logger.info("Final Score = 0.5*AUC_clean + 0.5*AUC_robust = %.4f", final_score)
        df.attrs["final_score"] = final_score

    return df


def main():
    parser = argparse.ArgumentParser(description="Build the robustness evaluation table.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/robustness_table.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    df = build_robustness_table(config, args.checkpoint, args.device)

    ensure_dir(str(Path(args.out).parent))
    df.to_csv(args.out, index=False)
    logger.info("Robustness table saved to %s", args.out)
    if "final_score" in df.attrs:
        with open(args.out.replace(".csv", "_final_score.txt"), "w") as f:
            f.write(f"Final Score: {df.attrs['final_score']:.4f}\n")


if __name__ == "__main__":
    main()
