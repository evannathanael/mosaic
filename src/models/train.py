"""Main training loop.

Person A: run with no --experiment flag for the main/baseline model.
Person B: run with --experiment <name> to test variants defined in
          configs/config.yaml -> experiments: (e.g. rotation_jitter, finetune_backbone).

Usage:
    python src/models/train.py --config configs/config.yaml
    python src/models/train.py --config configs/config.yaml --experiment rotation_jitter
"""
import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data.dataset import scan_dataset, split_samples, AIGCDataset
from src.models.classifier import AIGCDetector
from src.utils import load_config, set_seed, get_logger, ensure_dir

logger = get_logger(__name__)


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float().to(device)
            labels = batch["label"].numpy()
            probs = model.predict_proba(images).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    return {
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan"),
        "accuracy": accuracy_score(all_labels, preds),
        "n_samples": len(all_labels),
    }


def train(config: dict):
    set_seed(config["seed"])
    device = config["training"]["device"] if torch.cuda.is_available() else "cpu"
    run_name = config["run_name"]
    logger.info("Starting training run: %s (device=%s)", run_name, device)

    # --- Data ---
    samples = scan_dataset(config["data"]["raw_dir"])
    if not samples:
        logger.error(
            "No samples found in %s — run src/data/download.py first.",
            config["data"]["raw_dir"],
        )
        return
    splits = split_samples(
        samples,
        holdout_generator=config["data"]["holdout_generator"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        seed=config["seed"],
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d, unseen_generator (holdout): %d",
        len(splits["train"]), len(splits["val"]), len(splits["test"]), len(splits["unseen_generator"]),
    )

    train_ds = AIGCDataset(splits["train"], config, mode="train")
    val_ds = AIGCDataset(splits["val"], config, mode="eval", eval_transform_name="clean")

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    # --- Model ---
    model = AIGCDetector(config).to(device)

    # Separate param groups so an unfrozen backbone (finetune_backbone
    # experiment) trains at a much smaller LR than the head — reusing the
    # head's LR for pretrained CLIP weights risks blowing them up. When the
    # backbone is fully frozen this group is just empty and AdamW ignores it.
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    logger.info(
        "Trainable parameters: %d head + %d backbone",
        sum(p.numel() for p in head_params), sum(p.numel() for p in backbone_params),
    )

    param_groups = [{"params": head_params, "lr": config["training"]["learning_rate"]}]
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": config["training"].get("backbone_learning_rate", config["training"]["learning_rate"]),
        })
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    criterion = nn.BCEWithLogitsLoss()

    # --- Train loop with early stopping ---
    best_auc = -1.0
    patience_counter = 0
    run_dir = ensure_dir(f"{config['output']['dir']}/{run_name}")

    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            images = batch["image"].float().to(device)
            labels = batch["label"].float().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d/%d - loss: %.4f - val_auc: %.4f - val_acc: %.4f - %.1fs",
            epoch + 1, config["training"]["epochs"], epoch_loss / len(train_loader),
            val_metrics["auc"], val_metrics["accuracy"], time.time() - t0,
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            model.save(str(run_dir / "model_best.pt"))
            logger.info("New best model saved (val_auc=%.4f)", best_auc)
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Save run metadata for reproducibility
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump({"best_val_auc": best_auc}, f, indent=2)

    logger.info("Training complete. Best val AUC: %.4f. Checkpoint: %s", best_auc, run_dir / "model_best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train the AIGC classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", default=None, help="Named experiment from config.yaml -> experiments:")
    args = parser.parse_args()

    config = load_config(args.config, experiment=args.experiment)
    train(config)


if __name__ == "__main__":
    main()
