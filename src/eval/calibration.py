"""Temperature scaling — makes the model's confidence scores meaningful
(e.g. "70% confident" actually means ~70% of such predictions are correct),
rather than saturated near 0% / 100%. Required since the deliverable JSON
explicitly asks for a confidence score ("pred"), not just a label.

Fits a single scalar temperature T on the validation set by minimizing NLL,
then bakes it into model.temperature so predict_proba() is calibrated
automatically afterwards.

Usage:
    python src/eval/calibration.py --checkpoint outputs/baseline/model_best.pt
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import split_samples
from src.eval.data_compat import scan_all_sources, load_eval_model, EvalDataset
from src.utils import load_config, get_logger

logger = get_logger(__name__)


def fit_temperature(model, val_loader: DataLoader, device: str, max_iter: int = 50) -> float:
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].float().to(device)
            embedding = model.embed(images)
            logits = model.logits_from_embedding(embedding)  # RAW logits, pre-temperature
            logits_list.append(logits)
            labels_list.append(batch["label"].float().to(device))

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_temp = temperature.item()
    logger.info("Fitted temperature: %.4f", final_temp)
    return final_temp


def main():
    parser = argparse.ArgumentParser(description="Calibrate model confidence via temperature scaling.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
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
    val_ds = EvalDataset(splits["val"], config, eval_transform_name="clean")
    val_loader = DataLoader(
        val_ds, batch_size=config["evaluation"].get("batch_size", config["training"]["batch_size"]), shuffle=False,
        num_workers=0, pin_memory=(args.device == "cuda"),
    )

    temperature = fit_temperature(model, val_loader, args.device)
    model.temperature.data = torch.tensor([temperature])
    model.save(args.checkpoint)  # overwrite checkpoint with calibrated temperature baked in
    logger.info("Calibrated checkpoint saved back to %s", args.checkpoint)


if __name__ == "__main__":
    main()
