"""Standalone SID-Set robustness eval: load a saved head checkpoint + the
frozen CLIP backbone, run ONLY against the validation split, report accuracy
per perturbation. No training here — train_clip_head.py already produced
outputs/sid_set_head.pt; this script just re-evaluates it.

Reuses train_clip_head.py's building blocks directly (embed_indices,
evaluate_head, load_head_checkpoint, SidSetParquetDataset, build_eval_transform,
PERTURBATIONS, stratified_indices) instead of reimplementing them, so this
eval is guaranteed to measure the same thing train_clip_head.py itself does.

Why this file exists (Colab memory fix): the combined train+eval script was
hanging/crashing on Colab partway through "Evaluating..." — 6 sequential
embedding passes over the full validation split, each spinning up DataLoader
worker processes and letting the previous condition's GPU tensors sit around
while the next one's embedding pass ran. This script:
  - forces NUM_WORKERS=0 (Colab notebooks are a known-flaky environment for
    multiprocessing DataLoader workers — this was the likely hang)
  - uses a small embedding batch size (8) to cut peak memory per batch
  - wraps embedding in torch.no_grad() — satisfied for free by reusing
    embed_indices(), which is already @torch.no_grad()-decorated
  - explicitly frees each condition's tensors (del + torch.cuda.empty_cache()
    + gc.collect()) before starting the next one, instead of letting 6
    conditions' worth of tensors accumulate across the loop

Usage:
    source venv/bin/activate
    python -m src.models.eval_only
"""
from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import torch

import src.models.train_clip_head as tch
from src.models.clip_aigc_head import AIGCClipDetector

# =====================================================================
# CONFIG
# =====================================================================
CHECKPOINT_PATH = Path("outputs/sid_set_head.pt")
EVAL_SUBSET = None  # images PER CLASS from the validation split (None = the whole split)

OUTPUT_DIR = Path("outputs")
TABLE_PATH = OUTPUT_DIR / "sid_set_clip_robustness_table.csv"

# --- Colab memory-safety overrides -----------------------------------
# embed_indices() (reused below, unmodified) reads NUM_WORKERS/EMBED_BATCH_SIZE
# as train_clip_head's own module globals at call time, so setting them here
# on the imported module is how this script tunes it without touching that
# function at all.
tch.NUM_WORKERS = 0
tch.EMBED_BATCH_SIZE = 8


def main() -> None:
    tch.set_seed(tch.SEED)
    device = tch.DEVICE
    print(f"Device: {device}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT_PATH} — run train_clip_head.py (DATASET='sid_set') first."
        )

    head, meta = tch.load_head_checkpoint(CHECKPOINT_PATH, device=device)
    print(f"Loaded head checkpoint: {meta}")

    print(f"Loading CLIP backbone: {meta['clip_model_name']} ({meta['clip_pretrained']})...")
    model = AIGCClipDetector(
        clip_model_name=meta["clip_model_name"],
        clip_pretrained=meta["clip_pretrained"],
        hidden_dim=meta["hidden_dim"],
        dropout=meta["dropout"],
    ).to(device)
    model.head.load_state_dict(head.state_dict())
    model.head.eval()

    clip_preprocess = model.clip_preprocess

    # --- ONLY the validation split ---
    manifest_path = tch.SID_SET_MANIFEST_DIR / "clean_manifest.csv"
    eval_base = tch.SidSetParquetDataset(
        manifest_path, tch.SID_SET_RAW_ROOT, split=tch.SID_SET_SPLITS["eval"], transform=None
    )
    eval_indices = tch.stratified_indices(eval_base, EVAL_SUBSET, seed=tch.SEED)
    print(f"Eval samples: {len(eval_indices)} (from {manifest_path}, split={tch.SID_SET_SPLITS['eval']!r})")

    # --- eval: clean, then each perturbation, each embedded once, freeing
    # memory between conditions so nothing accumulates across the loop ---
    print("\nEvaluating (each condition embedded once)...")
    results = []
    conditions = {"clean": None, **tch.PERTURBATIONS}
    for name, perturbation_fn in conditions.items():
        tch.set_seed(tch.SEED)  # keep perturbations with randomness (RRC, color jitter) reproducible run-to-run
        eval_features, eval_labels = tch.embed_indices(
            model, eval_base, eval_indices, tch.build_eval_transform(clip_preprocess, perturbation_fn), desc=name
        )
        acc = tch.evaluate_head(model, eval_features, eval_labels)
        results.append({"condition": name, "n": len(eval_indices), "accuracy": round(acc, 4)})
        print(f"  {name:<22} accuracy: {acc:.4f}")

        # Free this condition's tensors before the next embedding pass starts
        # — this is the step the original combined script skipped, letting
        # GPU memory pile up across all 6 conditions until Colab hung/OOM'd.
        del eval_features, eval_labels
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    table = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)

    print("\n=== Robustness table (clean vs. each perturbation) ===")
    print(table.to_string(index=False))
    print(f"\nSaved to {TABLE_PATH}")


if __name__ == "__main__":
    main()
