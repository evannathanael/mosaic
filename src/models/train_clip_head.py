"""Train a frozen-CLIP + MLP-head AI-image detector on CIFAKE, then report
accuracy on clean test images AND on each robustness perturbation separately.

Pipeline:
    1. Load CIFAKE train/ and test/ with torchvision.datasets.ImageFolder
       (each has REAL/ and FAKE/ subfolders).
    2. Wrap the labels so 0 = real, 1 = AI ("fake") — ImageFolder assigns
       indices alphabetically (FAKE=0, REAL=1), which is the OPPOSITE of the
       convention this project uses, so BinaryCifakeFolder remaps it.
    3. Build the model: frozen open_clip ViT-L/14 backbone (`AIGCClipDetector`
       in clip_aigc_head.py) + a small trainable MLP head.
    4. Train ONLY the head with images passed through CLIP's own preprocessing,
       with training-time robustness augmentations (JPEG recompression,
       Gaussian blur, random-resized crop, downscale-then-upscale, color
       jitter) applied beforehand so the head learns to be robust to them.
    5. Evaluate: run the eval subset through CLIP preprocessing alone
       ("clean"), then once more per perturbation in isolation, and print +
       save a table of accuracy per condition — the key deliverable.

Quick CPU debugging: TRAIN_SUBSET / VAL_SUBSET below cap how many images (per
class) get used, so you can run the whole pipeline end-to-end in well under a
minute on a laptop with no GPU before scaling up on Colab. Set both to None
to use the full CIFAKE train/test split.

Usage:
    source venv/bin/activate
    python -m src.models.train_clip_head
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.models.clip_aigc_head import AIGCClipDetector

# =====================================================================
# CONFIG — edit these first. Shrink the *_SUBSET values for fast CPU runs;
# set them to None to use the full dataset once you scale up (e.g. on Colab).
# =====================================================================
TRAIN_SUBSET = 200   # images PER CLASS from data/raw/CIFAKE/train (None = all)
VAL_SUBSET = 200      # images PER CLASS from data/raw/CIFAKE/test, used for eval/the table (None = all)

# CIFAKE lives at data/raw/CIFAKE/{train,test}/{REAL,FAKE} in this repo (see
# data/README.md) — not ./data/train and ./data/test. Point this at wherever
# your copy actually is; the rest of the script only cares that DATA_ROOT has
# train/ and test/ subfolders, each with REAL/ and FAKE/.
DATA_ROOT = Path("data/raw/CIFAKE")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"   # first run downloads + caches weights (~890MB) via open_clip/HF hub
HEAD_HIDDEN_DIM = 256
HEAD_DROPOUT = 0.2

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# CPU-only on purpose: MPS exists on Apple Silicon but some open_clip ops
# still fall back to CPU (or error) on it inconsistently across torch
# versions, which is exactly the kind of flakiness you don't want while
# debugging on a small subset. Get it running clean on CPU first; on Colab
# this will pick up CUDA automatically further down.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
NUM_WORKERS = 0  # keep 0 on macOS for this debug script — avoids fork overhead on tiny subsets

OUTPUT_DIR = Path("outputs")
TABLE_PATH = OUTPUT_DIR / "cifake_clip_robustness_table.csv"

# --- robustness perturbation severities (used for both train-time random
# augmentation and the fixed eval perturbations below) ---
JPEG_QUALITY = 30
BLUR_RADIUS = 2.0
RRC_SCALE = (0.5, 1.0)
DOWNSCALE_FACTOR = 0.25
COLOR_JITTER_STRENGTH = 0.3
TRAIN_AUG_PROB = 0.5  # probability each individual perturbation is applied during training


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


# =====================================================================
# Step 1-2: data — ImageFolder + label remap so 0=real, 1=fake
# =====================================================================
class BinaryCifakeFolder(ImageFolder):
    """ImageFolder over a REAL/FAKE directory, relabeled to 0=real, 1=fake.

    Plain ImageFolder assigns class indices alphabetically, so REAL and FAKE
    map to 1 and 0 respectively — backwards from this project's convention.
    We build the correct remap once from self.classes (whatever order/case
    they're in) and apply it in __getitem__.
    """

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        name_to_binary = {"real": 0, "fake": 1}
        try:
            self.remap = [name_to_binary[name.lower()] for name in self.classes]
        except KeyError as exc:
            raise ValueError(
                f"Expected REAL/FAKE class folders under {root}, found {self.classes}"
            ) from exc
        # relabel everywhere torchvision/callers might read targets from
        self.targets = [self.remap[t] for t in self.targets]
        self.samples = [(path, self.remap[t]) for path, t in self.samples]
        self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def stratified_subset(dataset: BinaryCifakeFolder, per_class: int | None, seed: int) -> Subset:
    """Balanced subset: up to `per_class` images of each label, so accuracy
    stays meaningful even at TRAIN_SUBSET/VAL_SUBSET=200.
    """
    if per_class is None:
        return Subset(dataset, list(range(len(dataset))))

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)

    indices: list[int] = []
    for label, idxs in by_class.items():
        rng.shuffle(idxs)
        indices.extend(idxs[:per_class])
    rng.shuffle(indices)
    return Subset(dataset, indices)


# =====================================================================
# Step 4a: robustness perturbations (operate on a PIL RGB image)
# =====================================================================
def jpeg_recompress(image: Image.Image, quality: int = JPEG_QUALITY) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(image: Image.Image, radius: float = BLUR_RADIUS) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def random_resized_crop(image: Image.Image, scale=RRC_SCALE) -> Image.Image:
    w, h = image.size
    return transforms.RandomResizedCrop(size=(h, w), scale=scale)(image)


def downscale_upscale(image: Image.Image, factor: float = DOWNSCALE_FACTOR) -> Image.Image:
    w, h = image.size
    small = image.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def color_jitter(image: Image.Image, strength: float = COLOR_JITTER_STRENGTH) -> Image.Image:
    return transforms.ColorJitter(brightness=strength, contrast=strength, saturation=strength)(image)


PERTURBATIONS = {
    "jpeg_recompress": jpeg_recompress,
    "gaussian_blur": gaussian_blur,
    "random_resized_crop": random_resized_crop,
    "downscale_upscale": downscale_upscale,
    "color_jitter": color_jitter,
}


class RandomRobustnessAugment:
    """Training-time transform: independently applies each perturbation with
    probability TRAIN_AUG_PROB, in a fixed order. Runs on a PIL image, before
    CLIP's own preprocessing.
    """

    def __init__(self, prob: float = TRAIN_AUG_PROB):
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        for fn in PERTURBATIONS.values():
            if random.random() < self.prob:
                image = fn(image)
        return image


def build_train_transform(clip_preprocess) -> transforms.Compose:
    return transforms.Compose([RandomRobustnessAugment(), clip_preprocess])


def build_eval_transform(clip_preprocess, perturbation_fn=None) -> transforms.Compose:
    steps = [perturbation_fn] if perturbation_fn is not None else []
    steps.append(clip_preprocess)
    return transforms.Compose(steps)


# =====================================================================
# Train + eval loops
# =====================================================================
def train_one_epoch(model: AIGCClipDetector, loader: DataLoader, optimizer, criterion) -> float:
    model.head.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: AIGCClipDetector, loader: DataLoader) -> float:
    model.head.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        probs = torch.sigmoid(model(images))
        preds = (probs >= 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else float("nan")


def main() -> None:
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    print(f"Loading CLIP backbone: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED}) — first run downloads weights...")
    model = AIGCClipDetector(
        clip_model_name=CLIP_MODEL_NAME,
        clip_pretrained=CLIP_PRETRAINED,
        hidden_dim=HEAD_HIDDEN_DIM,
        dropout=HEAD_DROPOUT,
    ).to(DEVICE)
    counts = model.param_counts()
    print(f"Backbone params (frozen): {counts['backbone']:,} | Head params (trainable): {counts['head']:,}")

    clip_preprocess = model.clip_preprocess

    # --- Step 1-2: datasets ---
    train_base = BinaryCifakeFolder(str(TRAIN_DIR), transform=build_train_transform(clip_preprocess))
    train_subset = stratified_subset(train_base, TRAIN_SUBSET, seed=SEED)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"Train samples: {len(train_subset)} (from {TRAIN_DIR})")

    # Eval dataset built WITHOUT a transform yet — each perturbation pass
    # below sets test_base.transform before iterating, reusing the same
    # underlying files/subset indices instead of re-scanning the directory.
    test_base = BinaryCifakeFolder(str(TEST_DIR), transform=None)
    eval_subset = stratified_subset(test_base, VAL_SUBSET, seed=SEED)
    print(f"Eval samples: {len(eval_subset)} (from {TEST_DIR})")

    # --- Step 4: train only the head ---
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch}/{EPOCHS} — train loss: {avg_loss:.4f}")

    # --- Step 5: eval — clean, then each perturbation in isolation ---
    print("\nEvaluating...")
    results = []
    conditions = {"clean": None, **PERTURBATIONS}
    for name, perturbation_fn in conditions.items():
        set_seed(SEED)  # keep perturbations with randomness (RRC, color jitter) reproducible run-to-run
        test_base.transform = build_eval_transform(clip_preprocess, perturbation_fn)
        eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        acc = evaluate(model, eval_loader)
        results.append({"condition": name, "n": len(eval_subset), "accuracy": round(acc, 4)})
        print(f"  {name:<22} accuracy: {acc:.4f}")

    table = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)

    print("\n=== Robustness table (clean vs. each perturbation) ===")
    print(table.to_string(index=False))
    print(f"\nSaved to {TABLE_PATH}")


if __name__ == "__main__":
    main()
