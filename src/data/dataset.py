"""Dataset class + train/val/test splitting logic.

Expected raw data layout (after download.py):

    data/raw/
        sid_set/{real,ai}/...
        cifake/{real,ai}/...
        wildfake/{real,ai}/<generator_name>/...   # generator name in subfolder

Splitting rules implemented here (important — read before changing):
  1. Splits are done at the SOURCE-IMAGE level, before any augmentation, so
     augmented/duplicated versions of the same image never leak across splits.
  2. One entire generator (config.data.holdout_generator) is excluded from
     train/val entirely and reserved for the "unseen generator" robustness
     test — this is what lets us report genuine generalization, not memorization.
"""
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import random_train_transform


@dataclass
class Sample:
    path: str
    label: int          # 0 = real, 1 = AI-generated
    generator: str       # e.g. "sid_set", "cifake", "wildfake_gan" — used for holdout logic


def scan_dataset(raw_dir: str) -> list[Sample]:
    """Walk data/raw/<dataset>/<real|ai>/... and build a flat sample list.
    For wildfake, the generator subfolder name is preserved (e.g.
    wildfake/ai/<generator_name>/img.png -> generator = 'wildfake_<generator_name>').
    """
    raw_dir = Path(raw_dir)
    samples: list[Sample] = []
    for dataset_dir in raw_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for label_name, label in (("real", 0), ("ai", 1)):
            label_dir = dataset_dir / label_name
            if not label_dir.exists():
                continue
            for img_path in label_dir.rglob("*"):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                if label == 1 and dataset_dir.name == "wildfake":
                    # preserve generator subfolder as part of the generator tag
                    rel = img_path.relative_to(label_dir)
                    generator = f"wildfake_{rel.parts[0]}" if len(rel.parts) > 1 else "wildfake_unknown"
                else:
                    generator = dataset_dir.name
                samples.append(Sample(path=str(img_path), label=label, generator=generator))
    return samples


def split_samples(
    samples: list[Sample],
    holdout_generator: str,
    train_split: float,
    val_split: float,
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """Leakage-free split at the source-image level, with one generator held
    out entirely for the unseen-generator generalization test.
    """
    rng = random.Random(seed)

    holdout = [s for s in samples if s.generator == holdout_generator]
    trainable_pool = [s for s in samples if s.generator != holdout_generator]

    rng.shuffle(trainable_pool)
    n = len(trainable_pool)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    return {
        "train": trainable_pool[:n_train],
        "val": trainable_pool[n_train:n_train + n_val],
        "test": trainable_pool[n_train + n_val:],
        "unseen_generator": holdout,
    }


class AIGCDataset(Dataset):
    """Loads images and applies either random robustness augmentation (train)
    or a fixed named eval transform (val/test), then returns (image, label).
    """

    def __init__(self, samples: list[Sample], config: dict, mode: str = "train", eval_transform_name: str = "clean"):
        self.samples = samples
        self.config = config
        self.mode = mode  # "train" or "eval"
        self.eval_transform_name = eval_transform_name
        self.image_size = config["data"]["image_size"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = np.array(Image.open(sample.path).convert("RGB"))

        if self.mode == "train":
            img = random_train_transform(img, self.config)
        else:
            from src.data.transforms import named_eval_transform
            img = named_eval_transform(self.eval_transform_name, img)

        img = Image.fromarray(img).resize((self.image_size, self.image_size))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5  # normalize to [-1, 1], matches CLIP preprocessing roughly

        return {
            "image": img_array.transpose(2, 0, 1),  # HWC -> CHW
            "label": sample.label,
            "path": sample.path,
            "generator": sample.generator,
        }
