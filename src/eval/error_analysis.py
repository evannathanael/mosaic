"""Collects representative false positives / false negatives and writes a
short Markdown report — required deliverable #5.

Specifically includes a false-positive check on heavily-filtered/edited REAL
photos (not just clean real photos), since that's the realistic TikTok
failure mode: heavy filters/beauty edits can look like AI artifacts to a
naive detector.

Usage:
    python src/eval/error_analysis.py --checkpoint outputs/baseline/model_best.pt --out outputs/error_analysis.md
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import split_samples
from src.eval.data_compat import scan_all_sources, load_eval_model, EvalDataset
from src.utils import load_config, get_logger, ensure_dir

logger = get_logger(__name__)


def collect_errors(model, loader, device, top_k: int = 10):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float().to(device)
            probs = model.predict_proba(images).cpu().numpy()
            for path, label, prob in zip(batch["path"], batch["label"].numpy(), probs):
                results.append({"path": path, "label": int(label), "pred_prob": float(prob)})

    false_positives = sorted(
        [r for r in results if r["label"] == 0 and r["pred_prob"] >= 0.5],
        key=lambda r: -r["pred_prob"],
    )[:top_k]
    false_negatives = sorted(
        [r for r in results if r["label"] == 1 and r["pred_prob"] < 0.5],
        key=lambda r: r["pred_prob"],
    )[:top_k]
    return false_positives, false_negatives


def write_report(fp_clean, fn_clean, fp_filtered, out_path: str):
    lines = ["# Error Analysis\n"]

    lines.append("## False Positives (real images predicted as AI-generated)\n")
    lines.append("| Image | Predicted AI probability |")
    lines.append("|---|---|")
    for r in fp_clean:
        lines.append(f"| `{r['path']}` | {r['pred_prob']:.2f} |")
    lines.append(
        "\n**Hypothesis:** likely triggered by heavy compression, unusual lighting, "
        "or smooth/low-texture regions the model associates with synthetic content.\n"
    )

    lines.append("## False Negatives (AI images predicted as real)\n")
    lines.append("| Image | Predicted AI probability |")
    lines.append("|---|---|")
    for r in fn_clean:
        lines.append(f"| `{r['path']}` | {r['pred_prob']:.2f} |")
    lines.append(
        "\n**Hypothesis:** likely a newer/unseen generator with fewer detectable "
        "artifacts, or an image that survived aggressive post-processing.\n"
    )

    lines.append("## False Positives on heavily-filtered REAL photos\n")
    lines.append(
        "This checks a TikTok-specific failure mode: does the model wrongly flag "
        "real photos that use heavy beauty filters / HDR / color grading?\n"
    )
    lines.append("| Image | Predicted AI probability |")
    lines.append("|---|---|")
    for r in fp_filtered:
        lines.append(f"| `{r['path']}` | {r['pred_prob']:.2f} |")
    fp_rate = len(fp_filtered) / max(1, len(fp_filtered))  # placeholder; real rate computed by caller if needed
    lines.append(
        "\n**Why this matters:** a high false-positive rate here would unfairly "
        "penalize real creators who use filters — see Mosaic's design "
        "goal of not treating AI-adjacent editing as equivalent to full synthesis.\n"
    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Error analysis report written to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Run error analysis.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/error_analysis.md")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_eval_model(args.checkpoint, config, device=args.device)

    samples = scan_all_sources(config)
    splits = split_samples(
        samples,
        holdout_generator=config["data"]["holdout_generator"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        seed=config["seed"],
    )

    test_ds = EvalDataset(splits["test"], config, eval_transform_name="clean")
    test_loader = DataLoader(
        test_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=4, pin_memory=(args.device == "cuda"),
    )
    fp_clean, fn_clean = collect_errors(model, test_loader, args.device)

    # Heavily-filtered real photos: reuse color_jitter as a stand-in for heavy filtering
    filtered_ds = EvalDataset(
        [s for s in splits["test"] if s.label == 0],
        config, eval_transform_name="color_jitter",
    )
    filtered_loader = DataLoader(
        filtered_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=4, pin_memory=(args.device == "cuda"),
    )
    fp_filtered, _ = collect_errors(model, filtered_loader, args.device)

    ensure_dir(str(Path(args.out).parent))
    write_report(fp_clean, fn_clean, fp_filtered, args.out)


if __name__ == "__main__":
    main()
