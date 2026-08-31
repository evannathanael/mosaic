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
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import random_train_transform

DEFAULT_SPLIT_CACHE_PATH = "data/processed/split_cache.json"


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


def _samples_fingerprint(samples: list[Sample]) -> str:
    """Hash of the exact set of sample paths (order-independent), used to
    tell whether a cached split still matches the current data/raw/ contents.
    """
    digest = hashlib.sha256()
    for path in sorted(s.path for s in samples):
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def split_samples(
    samples: list[Sample],
    holdout_generator: str,
    train_split: float,
    val_split: float,
    seed: int = 42,
    cache_path: str | Path | None = DEFAULT_SPLIT_CACHE_PATH,
) -> dict[str, list[Sample]]:
    """Leakage-free split at the source-image level, with one generator held
    out entirely for the unseen-generator generalization test.

    Cached to `cache_path` (by default), keyed on
    (holdout_generator, train_split, val_split, seed, and a fingerprint of
    the exact sample paths). Without this, re-running training/eval against
    a fresh scan_dataset() scan is NOT guaranteed to reproduce the same
    split even with the same seed: filesystem enumeration order isn't
    stable, so the same seed shuffling a differently-ordered input list
    gives a different result — this was a verified, real source of two
    identical-code runs training/testing on different images. Once a split
    is cached for a given fingerprint, every later call with the same
    fingerprint/config reuses the exact same split; if the underlying
    sample set genuinely changes (files added/removed under data/raw/), the
    fingerprint changes too and the cache is recomputed and overwritten.
    Pass cache_path=None to always recompute (e.g. in tests).
    """
    cache_file = Path(cache_path) if cache_path else None
    cache_key = {
        "holdout_generator": holdout_generator,
        "train_split": train_split,
        "val_split": val_split,
        "seed": seed,
        "samples_fingerprint": _samples_fingerprint(samples),
    }

    if cache_file and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        if cached and cached.get("key") == cache_key:
            path_to_sample = {s.path: s for s in samples}
            return {
                split_name: [path_to_sample[p] for p in paths if p in path_to_sample]
                for split_name, paths in cached["splits"].items()
            }

    rng = random.Random(seed)

    holdout = [s for s in samples if s.generator == holdout_generator]
    trainable_pool = [s for s in samples if s.generator != holdout_generator]

    rng.shuffle(trainable_pool)
    n = len(trainable_pool)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    result = {
        "train": trainable_pool[:n_train],
        "val": trainable_pool[n_train:n_train + n_val],
        "test": trainable_pool[n_train + n_val:],
        "unseen_generator": holdout,
    }

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({
                "key": cache_key,
                "splits": {name: [s.path for s in split] for name, split in result.items()},
            }, indent=2),
            encoding="utf-8",
        )

    return result


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
