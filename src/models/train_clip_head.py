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
    4. EMBED ONCE, then train the head on cached embeddings (the perf fix):
       the backbone is frozen, so an image's embedding is fixed the moment
       you've decided what augmented version of it you're training on. There
       is no reason to re-run the ~430M-param backbone forward pass on every
       image on every epoch just to recompute the same vector. So: run every
       training image (after robustness augmentation) through the backbone
       ONE time, cache the resulting (embedding, label) tensors in memory,
       then loop epochs over those cached tensors — pure MLP forward/backward,
       no backbone involved. This is the standard "frozen-features" trick and
       turns epochs from minutes into seconds; only the one-time embedding
       pass is backbone-bound.
       Trade-off worth knowing: because each training image is embedded once,
       it gets ONE randomly-drawn augmentation for the whole run rather than a
       fresh one every epoch. That's what "cache the embeddings" necessarily
       means for a randomized augmentation pipeline — the augmentation logic
       itself (which perturbations, what probability) is unchanged.
    5. Evaluate: same idea. "Clean" and each perturbation condition are each
       a genuinely different image, so each needs its own embedding — but
       exactly once per condition (not once per epoch, since eval doesn't
       loop epochs), cached the same way, then scored against the trained
       head instantly.

Quick CPU debugging: TRAIN_SUBSET / VAL_SUBSET below cap how many images (per
class) get used, so you can run the whole pipeline end-to-end quickly on a
laptop with no GPU before scaling up on Colab. Set both to None to use the
full CIFAKE train/test split. On CUDA, embedding is the only backbone-bound
step and batches/parallel workers scale up automatically (see EMBED_BATCH_SIZE
/ NUM_WORKERS below).

Usage:
    source venv/bin/activate
    python -m src.models.train_clip_head
"""
from __future__ import annotations

import io
import random
import time
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

BATCH_SIZE = 16       # head-training batch size — trains on cached embeddings, so this is cheap regardless
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
# The one remaining backbone-bound step is embedding — batch it generously,
# and use a couple of DataLoader workers on CUDA (Colab) to keep the GPU fed
# with decoded/augmented images. Keep workers at 0 on macOS/CPU: with tiny
# debug subsets, process-fork overhead costs more than it saves.
EMBED_BATCH_SIZE = 64
NUM_WORKERS = 4 if DEVICE == "cuda" else 0
PROGRESS_EVERY = 1000  # print embedding progress every N images (full-scale runs only)

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


def stratified_indices(dataset: BinaryCifakeFolder, per_class: int | None, seed: int) -> list[int]:
    """Balanced index list: up to `per_class` images of each label, so
    accuracy stays meaningful even at TRAIN_SUBSET/VAL_SUBSET=200.
    """
    if per_class is None:
        return list(range(len(dataset)))

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)

    indices: list[int] = []
    for label, idxs in by_class.items():
        rng.shuffle(idxs)
        indices.extend(idxs[:per_class])
    rng.shuffle(indices)
    return indices


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
# Step 4b: embed once, cache — the actual perf fix
# =====================================================================
@torch.no_grad()
def embed_indices(
    model: AIGCClipDetector,
    base_dataset: BinaryCifakeFolder,
    indices: list[int],
    transform: transforms.Compose,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run every image in `indices` through the frozen backbone EXACTLY ONCE
    under the given transform, returning cached (features, labels) tensors on
    DEVICE. This is the one backbone-bound pass; everything downstream
    (training epochs, evaluation for this condition) reuses these tensors.
    """
    base_dataset.transform = transform
    subset = Subset(base_dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=EMBED_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    features, labels = [], []
    seen = 0
    t0 = time.time()
    for images, batch_labels in loader:
        images = images.to(DEVICE, non_blocking=(DEVICE == "cuda"))
        features.append(model.encode_image(images))
        labels.append(batch_labels)
        seen += images.size(0)
        if seen % PROGRESS_EVERY < EMBED_BATCH_SIZE or seen == len(subset):
            print(f"  embedding [{desc}]: {seen}/{len(subset)}", end="\r", flush=True)

    if len(subset):
        print(f"  embedding [{desc}]: {len(subset)}/{len(subset)} done in {time.time() - t0:.1f}s")
    features_t = torch.cat(features) if features else torch.empty(0, model.embed_dim, device=DEVICE)
    labels_t = torch.cat(labels).to(DEVICE) if labels else torch.empty(0, dtype=torch.long, device=DEVICE)
    return features_t, labels_t


def train_head(
    model: AIGCClipDetector,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> None:
    """Trains ONLY the MLP head on cached embeddings — no backbone calls, no
    image decoding, just small matmuls. This is the loop that used to take
    minutes per epoch; on cached features it takes seconds.
    """
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    n = features.size(0)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.head.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_features = features[idx]
            batch_labels = labels[idx].float()

            optimizer.zero_grad()
            logits = model.head(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_features.size(0)

        print(f"Epoch {epoch}/{epochs} — train loss: {total_loss / n:.4f}  ({time.time() - t0:.2f}s)")


@torch.no_grad()
def evaluate_head(model: AIGCClipDetector, features: torch.Tensor, labels: torch.Tensor) -> float:
    if features.size(0) == 0:
        return float("nan")
    model.head.eval()
    probs = torch.sigmoid(model.head(features))
    preds = (probs >= 0.5).long()
    return (preds == labels).float().mean().item()


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

    # --- Step 1-2: datasets (transform set per-embedding-pass below) ---
    train_base = BinaryCifakeFolder(str(TRAIN_DIR), transform=None)
    train_indices = stratified_indices(train_base, TRAIN_SUBSET, seed=SEED)
    print(f"Train samples: {len(train_indices)} (from {TRAIN_DIR})")

    test_base = BinaryCifakeFolder(str(TEST_DIR), transform=None)
    eval_indices = stratified_indices(test_base, VAL_SUBSET, seed=SEED)
    print(f"Eval samples: {len(eval_indices)} (from {TEST_DIR})")

    # --- Step 4b: embed the training set ONCE (backbone-bound; everything
    # after this is just MLP training on the cached tensors) ---
    print("\nEmbedding training set (one backbone pass, cached for all epochs)...")
    train_features, train_labels = embed_indices(
        model, train_base, train_indices, build_train_transform(clip_preprocess), desc="train"
    )

    # --- Step 4: train only the head, on cached embeddings ---
    print("\nTraining head on cached embeddings...")
    train_head(model, train_features, train_labels, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY)

    # --- Step 5: eval — clean, then each perturbation, each embedded once ---
    print("\nEvaluating (each condition embedded once)...")
    results = []
    conditions = {"clean": None, **PERTURBATIONS}
    for name, perturbation_fn in conditions.items():
        set_seed(SEED)  # keep perturbations with randomness (RRC, color jitter) reproducible run-to-run
        eval_features, eval_labels = embed_indices(
            model, test_base, eval_indices, build_eval_transform(clip_preprocess, perturbation_fn), desc=name
        )
        acc = evaluate_head(model, eval_features, eval_labels)
        results.append({"condition": name, "n": len(eval_indices), "accuracy": round(acc, 4)})
        print(f"  {name:<22} accuracy: {acc:.4f}")

    table = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)

    print("\n=== Robustness table (clean vs. each perturbation) ===")
    print(table.to_string(index=False))
    print(f"\nSaved to {TABLE_PATH}")


if __name__ == "__main__":
    main()
