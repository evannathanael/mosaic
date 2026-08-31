"""Train a frozen-CLIP + MLP-head AI-image detector on CIFAKE, SID-Set, or
WildFake, then report accuracy on clean test images AND on each robustness
perturbation separately.

Which dataset trains/evaluates is a single switch — DATASET below — so you
can flip between them without touching anything downstream:
    - "cifake":   torchvision.datasets.ImageFolder over data/raw/CIFAKE/{train,test}
    - "sid_set":  reads Vicky's cleaned manifest (data/processed/sid_set) and
      fetches image bytes on demand from the Parquet shards it references
      (data/raw/sid_set) — see SidSetParquetDataset below. SID-Set has 3
      source labels (0=real, 1=full synthetic, 2=tampered); this loader keeps
      0->0 and 1->1 and drops 2 entirely, applying that binary policy itself
      since Vicky's cleaner deliberately does not (confirmed in its docstring
      and data/README.md — see sid_set_label_policy()).
    - "wildfake": reads Vicky's cleaned manifest (data/processed/wildfake) and
      fetches image bytes on demand from the ZIP archives it references
      (data/raw/wildfake) — see WildfakeArchiveDataset below. UNLIKE SID-Set,
      src/data/clean_wildfake.py already writes BINARY source_label values (0
      real, 1 AI) straight from the archive source, so there's no 3-way
      policy to collapse here — see wildfake_label_policy(), which is an
      identity mapping kept only so every dataset has one obvious place its
      label policy lives.
      NOTE: WildFake is NOT loaded via the ModelScope SDK (no
      modelscope.msdatasets.MsDataset.load() anywhere in this repo) — it
      requires manually translating the dataset page and downloading a
      hand-picked archive subset first (src/data/download.py's
      download_wildfake() just prints that reminder, it doesn't fetch
      anything), then Vicky's clean_wildfake.py validates those ZIPs in
      place. This loader reads that already-established manifest+archive
      system rather than hitting ModelScope at runtime, matching how the
      SID-Set loader reads Vicky's manifest instead of re-touching Hugging
      Face at runtime.
Every loader produces the exact same (image, binary_label) shape per item, so
everything past dataset construction — the frozen backbone, the embedding
cache, the MLP head, robustness eval, and checkpointing — is 100% shared code.

    - "combined": trains ONE head jointly over all three sources (avoiding
      the catastrophic forgetting that sequential per-source fine-tuning
      causes with no rehearsal), by embedding each source's train split
      separately (each needs its own dataset/transform) and concatenating
      the resulting cached tensors before a single train_head() call — see
      run_combined() below. Evaluation reports accuracy per source AND
      pooled, for every condition, so a source-specific regression (like a
      model that's forgotten CIFAKE) is visible in the table rather than
      averaged away.

Pipeline:
    1. Load the selected dataset (see DATASET switch above).
    2. Labels are 0 = real, 1 = AI ("fake") for all three datasets. CIFAKE's
       ImageFolder assigns indices alphabetically (FAKE=0, REAL=1) — the
       OPPOSITE of this project's convention — so BinaryCifakeFolder remaps
       it; SID-Set's and WildFake's remaps are the label policies described
       above.
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

import csv
import io
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.models.clip_aigc_head import AIGCClipDetector, MlpHead

# =====================================================================
# CONFIG — edit these first. Shrink the *_SUBSET values for fast CPU runs;
# set them to None to use the full dataset once you scale up (e.g. on Colab).
# =====================================================================
DATASET = "cifake"  # "cifake", "sid_set", "wildfake", or "combined" — flip this to switch; also names outputs/{DATASET}_*
COMBINED_SOURCES = ["cifake", "sid_set", "wildfake"]  # pooled, in this order, when DATASET == "combined"

TRAIN_SUBSET = 200   # images PER CLASS from the selected dataset's train split (None = all)
VAL_SUBSET = 200      # images PER CLASS from the selected dataset's eval split, used for the table (None = all)

# CIFAKE lives at data/raw/CIFAKE/{train,test}/{REAL,FAKE} in this repo (see
# data/README.md) — not ./data/train and ./data/test. Point this at wherever
# your copy actually is; the rest of the script only cares that DATA_ROOT has
# train/ and test/ subfolders, each with REAL/ and FAKE/.
DATA_ROOT = Path("data/raw/CIFAKE")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

# SID-Set: Vicky's cleaner (src/data/clean_sid_set.py) writes a row-level
# manifest here, referencing each image by (parquet_file, row_index) rather
# than copying it out of Parquet — SID_SET_MANIFEST_DIR/clean_manifest.csv is
# what SidSetParquetDataset reads. SID_SET_RAW_ROOT is where the actual
# Parquet shards live: src/data/download.py's snapshot_download() places the
# HF repo at data/raw/sid_set, and HF dataset repos for SID-Set nest their
# shards under a data/ subfolder inside that (matching the manifest's
# recorded paths, e.g. "data/train-00000-of-00249.parquet") — so this should
# resolve correctly once you've actually run download.py. (Note: Vicky's own
# cleaning run used a differently-named local folder, sid_set_hf, per
# data/processed/sid_set/cleaning_report.json — that only affected where SHE
# ran the cleaner from, not the root-relative paths recorded in the manifest.)
SID_SET_RAW_ROOT = Path("data/raw/sid_set")
SID_SET_MANIFEST_DIR = Path("data/processed/sid_set")
SID_SET_SPLITS = {"train": "train", "eval": "validation"}  # SID-Set's eval split is named "validation", not "test"

# WildFake: Vicky's cleaner (src/data/clean_wildfake.py) writes a manifest
# here referencing each image by (archive_path, member_path) inside a ZIP —
# WILDFAKE_MANIFEST_DIR/clean_manifest.csv is what WildfakeArchiveDataset
# reads, WILDFAKE_RAW_ROOT is where the (manually downloaded) ZIP archives
# live. clean_wildfake.py's splits are train/validation/test; this loader
# uses "validation" as the eval split, matching SID_SET_SPLITS's convention.
WILDFAKE_RAW_ROOT = Path("data/raw/wildfake")
WILDFAKE_MANIFEST_DIR = Path("data/processed/wildfake")
WILDFAKE_SPLITS = {"train": "train", "eval": "validation"}

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
TABLE_PATH = OUTPUT_DIR / f"{DATASET}_clip_robustness_table.csv"
HEAD_CHECKPOINT_PATH = OUTPUT_DIR / f"{DATASET}_head.pt"

# --- robustness perturbation severities (used for both train-time random
# augmentation and the fixed eval perturbations below) ---
JPEG_QUALITY = 30
BLUR_RADIUS = 2.0
RRC_SCALE = (0.5, 1.0)
DOWNSCALE_FACTOR = 0.25
COLOR_JITTER_STRENGTH = 0.3
NOISE_SIGMA = 0.10  # matches config.yaml's noise_0.10 — the specific condition robustness.py flagged as weakest
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
# Step 1-2 (alt): SID-Set — Parquet-backed dataset, dataset-agnostic downstream
# =====================================================================
def sid_set_label_policy(source_label: int) -> int | None:
    """0=real -> 0, 1=full synthetic -> 1, 2=tampered -> None (excluded).

    Vicky's cleaner (src/data/clean_sid_set.py) explicitly does NOT apply a
    binary policy — its docstring says so verbatim ("No binary training-label
    policy is applied here"), and data/README.md confirms it's left as an
    intentional later step: "The binary mapping for model training is
    intentionally a later step." So clean_manifest.csv still carries the
    original 3-way source_label, and this is the one place the binary policy
    gets applied — do not re-apply it upstream in the cleaner.
    """
    if source_label in (0, 1):
        return source_label
    return None  # tampered — excluded entirely for now


def _extract_image_bytes(image_value) -> bytes:
    """Pull raw bytes out of a Parquet image struct value — mirrors
    src/data/clean_sid_set.py's `_image_bytes` helper (kept local, not
    imported, so this script stays one portable file).
    """
    if isinstance(image_value, dict):
        raw = image_value.get("bytes")
        if raw is None:
            raise ValueError("image.bytes is missing (row references an external file, not embedded bytes)")
        return raw
    if isinstance(image_value, (bytes, bytearray)):
        return bytes(image_value)
    raise ValueError(f"Unexpected image column value type: {type(image_value)}")


class SidSetParquetDataset(torch.utils.data.Dataset):
    """Reads SID-Set images referenced by Vicky's cleaned manifest, fetching
    image bytes on demand from the Parquet shards under `raw_root` — the
    manifest stores (parquet_file, row_index), not copies of the images.

    Duck-types the same interface BinaryCifakeFolder exposes — `.samples`
    (list of (identifier, binary_label)), a mutable `.transform`, and
    `__getitem__` -> (image, binary_label) — so stratified_indices() and
    embed_indices() work on it completely unchanged.
    """

    def __init__(self, manifest_path: Path, raw_root: Path, split: str, transform=None):
        self.raw_root = Path(raw_root)
        self.transform = transform
        self.samples: list[tuple[tuple[str, int], int]] = []

        with open(manifest_path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["split"] != split:
                    continue
                binary_label = sid_set_label_policy(int(row["source_label"]))
                if binary_label is None:
                    continue  # tampered (2) — excluded per the label policy above
                self.samples.append(((row["parquet_file"], int(row["row_index"])), binary_label))

        if not self.samples:
            raise ValueError(
                f"No usable rows for split={split!r} in {manifest_path} after the label policy — "
                f"check the manifest exists (run src/data/clean_sid_set.py) and that split name is right."
            )

        # parquet_file -> list of raw image-column values, loaded lazily and
        # cached per shard. On CUDA/Colab, DataLoader workers each get their
        # own copy of this dataset (spawned, not forked, on most platforms),
        # so each worker builds its own cache independently — a little
        # redundant I/O across workers, but no correctness issue.
        self._shard_cache: dict[str, list] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _shard_images(self, parquet_file: str) -> list:
        if parquet_file not in self._shard_cache:
            table = pq.read_table(self.raw_root / parquet_file, columns=["image"])
            self._shard_cache[parquet_file] = table.column("image").to_pylist()
        return self._shard_cache[parquet_file]

    def __getitem__(self, index: int):
        (parquet_file, row_index), label = self.samples[index]
        raw = _extract_image_bytes(self._shard_images(parquet_file)[row_index])
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# =====================================================================
# Step 1-2 (alt): WildFake — ZIP-archive-backed dataset, same role as
# SidSetParquetDataset, different on-demand storage format
# =====================================================================
def wildfake_label_policy(source_label: int) -> int | None:
    """0=real -> 0, 1=AI -> 1.

    Unlike SID-Set, src/data/clean_wildfake.py already writes BINARY
    source_label values straight from the archive source (0 real, 1 AI) —
    its docstring says so verbatim ("Labels are binary and are derived only
    from the archive source... no relabeling or model-based filtering is
    performed"), confirmed in data/README.md too. So this is an identity
    mapping — there's no 3-way scheme to collapse here — kept as its own
    function only so every dataset's label policy lives in one obvious,
    named place, matching sid_set_label_policy()'s pattern.
    """
    if source_label in (0, 1):
        return source_label
    return None


class WildfakeArchiveDataset(torch.utils.data.Dataset):
    """Reads WildFake images referenced by Vicky's cleaned manifest, fetching
    image bytes on demand from the ZIP archives under `raw_root` — the
    manifest stores (archive_path, member_path), not copies of the images.

    Same role as SidSetParquetDataset, different storage: ZIP files support
    genuinely cheap random-access reads of a single named member (no need to
    materialize a whole archive's images in memory the way a Parquet column
    read does), so this only caches open ZipFile handles, not decoded bytes.

    Duck-types the same interface — `.samples`, mutable `.transform`,
    `__getitem__` -> (image, binary_label) — so stratified_indices() and
    embed_indices() work on it completely unchanged.
    """

    def __init__(self, manifest_path: Path, raw_root: Path, split: str, transform=None):
        self.raw_root = Path(raw_root)
        self.transform = transform
        self.samples: list[tuple[tuple[str, str], int]] = []

        with open(manifest_path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["split"] != split:
                    continue
                binary_label = wildfake_label_policy(int(row["source_label"]))
                if binary_label is None:
                    continue
                self.samples.append(((row["archive_path"], row["member_path"]), binary_label))

        if not self.samples:
            raise ValueError(
                f"No usable rows for split={split!r} in {manifest_path} after the label policy — "
                f"check the manifest exists (run src/data/clean_wildfake.py) and that split name is right."
            )

        self._archive_cache: dict[str, zipfile.ZipFile] = {}  # archive_path -> open handle, per worker process

    def __len__(self) -> int:
        return len(self.samples)

    def _open_archive(self, archive_path: str) -> zipfile.ZipFile:
        if archive_path not in self._archive_cache:
            self._archive_cache[archive_path] = zipfile.ZipFile(self.raw_root / archive_path)
        return self._archive_cache[archive_path]

    def __getitem__(self, index: int):
        (archive_path, member_path), label = self.samples[index]
        raw = self._open_archive(archive_path).read(member_path)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# =====================================================================
# Dataset builders — the ONLY thing the DATASET switch changes. Everything
# past this point (embed_indices, train_head, evaluate_head, checkpointing)
# only ever sees (dataset, indices) and doesn't know or care which dataset
# produced them.
# =====================================================================
def build_cifake_datasets():
    train_base = BinaryCifakeFolder(str(TRAIN_DIR), transform=None)
    train_indices = stratified_indices(train_base, TRAIN_SUBSET, seed=SEED)
    eval_base = BinaryCifakeFolder(str(TEST_DIR), transform=None)
    eval_indices = stratified_indices(eval_base, VAL_SUBSET, seed=SEED)
    print(f"Train samples: {len(train_indices)} (from {TRAIN_DIR})")
    print(f"Eval samples: {len(eval_indices)} (from {TEST_DIR})")
    return train_base, train_indices, eval_base, eval_indices


def build_sid_set_datasets():
    manifest_path = SID_SET_MANIFEST_DIR / "clean_manifest.csv"
    train_base = SidSetParquetDataset(manifest_path, SID_SET_RAW_ROOT, split=SID_SET_SPLITS["train"], transform=None)
    train_indices = stratified_indices(train_base, TRAIN_SUBSET, seed=SEED)
    eval_base = SidSetParquetDataset(manifest_path, SID_SET_RAW_ROOT, split=SID_SET_SPLITS["eval"], transform=None)
    eval_indices = stratified_indices(eval_base, VAL_SUBSET, seed=SEED)
    print(f"Train samples: {len(train_indices)} (from {manifest_path}, split={SID_SET_SPLITS['train']!r})")
    print(f"Eval samples: {len(eval_indices)} (from {manifest_path}, split={SID_SET_SPLITS['eval']!r})")
    return train_base, train_indices, eval_base, eval_indices


def build_wildfake_datasets():
    manifest_path = WILDFAKE_MANIFEST_DIR / "clean_manifest.csv"
    train_base = WildfakeArchiveDataset(
        manifest_path, WILDFAKE_RAW_ROOT, split=WILDFAKE_SPLITS["train"], transform=None
    )
    train_indices = stratified_indices(train_base, TRAIN_SUBSET, seed=SEED)
    eval_base = WildfakeArchiveDataset(
        manifest_path, WILDFAKE_RAW_ROOT, split=WILDFAKE_SPLITS["eval"], transform=None
    )
    eval_indices = stratified_indices(eval_base, VAL_SUBSET, seed=SEED)
    print(f"Train samples: {len(train_indices)} (from {manifest_path}, split={WILDFAKE_SPLITS['train']!r})")
    print(f"Eval samples: {len(eval_indices)} (from {manifest_path}, split={WILDFAKE_SPLITS['eval']!r})")
    return train_base, train_indices, eval_base, eval_indices


DATASET_BUILDERS = {
    "cifake": build_cifake_datasets,
    "sid_set": build_sid_set_datasets,
    "wildfake": build_wildfake_datasets,
}


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


def gaussian_noise(image: Image.Image, sigma: float = NOISE_SIGMA) -> Image.Image:
    """Additive Gaussian pixel noise — same math as src/data/transforms.py's
    apply_gaussian_noise (sigma is a fraction of the 0-255 range), kept as a
    local reimplementation rather than an import so this script stays one
    portable file. This was the one condition config.yaml's official 6-
    transform grid requires that training augmentation never covered —
    added after robustness.py's full table showed it as the biggest AUC
    drop of the six (0.749 clean -> 0.662 at noise_0.10) on an
    otherwise-untrained-for perturbation.
    """
    arr = np.asarray(image).astype(np.float32)
    noise = np.random.normal(0, sigma * 255, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# PERTURBATIONS drives the eval table (conditions = {"clean": None,
# **PERTURBATIONS}, in main(), run_combined(), AND eval_only.py) — this set
# is intentionally left exactly as it was, so the eval table's rows are
# unchanged. gaussian_noise is deliberately NOT in here.
PERTURBATIONS = {
    "jpeg_recompress": jpeg_recompress,
    "gaussian_blur": gaussian_blur,
    "random_resized_crop": random_resized_crop,
    "downscale_upscale": downscale_upscale,
    "color_jitter": color_jitter,
}

# TRAIN_PERTURBATIONS is what RandomRobustnessAugment actually applies during
# training — PERTURBATIONS plus gaussian_noise. Kept as a separate dict
# (rather than adding noise straight into PERTURBATIONS) specifically so
# training coverage can include noise without changing what the eval table
# reports — training augmentation and eval conditions are two different
# concerns that happened to share one dict before this split, which is what
# caused eval to silently gain a 7th row when noise was added; don't
# recombine them without deciding that's actually wanted.
TRAIN_PERTURBATIONS = {**PERTURBATIONS, "gaussian_noise": gaussian_noise}


class RandomRobustnessAugment:
    """Training-time transform: independently applies each perturbation with
    probability TRAIN_AUG_PROB, in a fixed order. Runs on a PIL image, before
    CLIP's own preprocessing.
    """

    def __init__(self, prob: float = TRAIN_AUG_PROB):
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        for fn in TRAIN_PERTURBATIONS.values():
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


# =====================================================================
# Checkpointing — save/load ONLY the trained head. The frozen CLIP backbone
# is never saved here: it's pretrained, unmodified, and cheaply re-obtained
# from (CLIP_MODEL_NAME, CLIP_PRETRAINED) via open_clip's own cache, so
# re-saving ~890MB of unchanged weights on every run would be pure waste.
# =====================================================================
def save_head_checkpoint(model: AIGCClipDetector, path: Path) -> None:
    """Save the trained head's weights + the config needed to reconstruct it
    (embed_dim/hidden_dim/dropout, and which CLIP backbone it was trained
    against), so a fresh process can reload it without retraining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": model.head.state_dict(),
            "embed_dim": model.embed_dim,
            "hidden_dim": HEAD_HIDDEN_DIM,
            "dropout": HEAD_DROPOUT,
            "clip_model_name": CLIP_MODEL_NAME,
            "clip_pretrained": CLIP_PRETRAINED,
        },
        path,
    )
    print(f"Saved head checkpoint to {path}")


def load_head_checkpoint(path: Path, device: str = DEVICE) -> tuple[MlpHead, dict]:
    """Load a saved head checkpoint for inference/evaluation.

    Returns (head, metadata) — metadata records which CLIP backbone this head
    was trained against (clip_model_name/clip_pretrained/embed_dim), so a
    caller can check it matches the backbone they pair it with. Usage:

        head, meta = load_head_checkpoint(Path("outputs/cifake_head.pt"))
        model = AIGCClipDetector(meta["clip_model_name"], meta["clip_pretrained"],
                                  meta["hidden_dim"], meta["dropout"])
        model.head.load_state_dict(head.state_dict())
    """
    checkpoint = torch.load(path, map_location=device)
    head = MlpHead(
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    )
    head.load_state_dict(checkpoint["head_state_dict"])
    head.to(device)
    head.eval()
    metadata = {k: v for k, v in checkpoint.items() if k != "head_state_dict"}
    return head, metadata


# =====================================================================
# Combined/joint training — pools cached embeddings from all COMBINED_SOURCES
# into one train_head() call, instead of separate per-source runs (which,
# run sequentially against a checkpoint that carries forward, is exactly the
# recipe for catastrophic forgetting: each stage's gradient updates have no
# way to "remember" the previous stage's data, since the frozen backbone
# gives a fixed feature space but the small head has no rehearsal signal).
# One head, one loss, gradients from every source in the same batches.
# =====================================================================
def run_combined(model: AIGCClipDetector, clip_preprocess) -> None:
    print(f"Combined training over sources: {COMBINED_SOURCES}")

    train_features_by_source: dict[str, torch.Tensor] = {}
    train_labels_by_source: dict[str, torch.Tensor] = {}
    eval_bases: dict[str, object] = {}
    eval_indices_by_source: dict[str, list[int]] = {}

    for source in COMBINED_SOURCES:
        print(f"\n--- {source} ---")
        train_base, train_indices, eval_base, eval_indices = DATASET_BUILDERS[source]()
        features, labels = embed_indices(
            model, train_base, train_indices, build_train_transform(clip_preprocess), desc=f"train:{source}"
        )
        train_features_by_source[source] = features
        train_labels_by_source[source] = labels
        eval_bases[source] = eval_base
        eval_indices_by_source[source] = eval_indices

    train_features = torch.cat(list(train_features_by_source.values()))
    train_labels = torch.cat(list(train_labels_by_source.values()))
    print(f"\nPooled train samples: {train_features.size(0)} across {len(COMBINED_SOURCES)} sources")

    print("\nTraining head jointly on pooled cached embeddings...")
    train_head(model, train_features, train_labels, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY)

    # --- Checkpoint: save the trained head, then verify it reloads correctly ---
    save_head_checkpoint(model, HEAD_CHECKPOINT_PATH)
    reloaded_head, checkpoint_meta = load_head_checkpoint(HEAD_CHECKPOINT_PATH, device=DEVICE)
    model.head.eval()
    with torch.no_grad():
        probe = train_features[: min(8, train_features.size(0))]
        original_out = model.head(probe)
        reloaded_out = reloaded_head(probe)
    if not torch.allclose(original_out, reloaded_out):
        raise RuntimeError("Checkpoint round-trip mismatch: reloaded head does not match the trained head.")
    print(f"Checkpoint verified: reloaded head matches trained weights (meta={checkpoint_meta}).")

    # --- Eval: every condition, scored per source AND pooled across sources ---
    print("\nEvaluating per source AND pooled (each condition embedded once per source)...")
    results = []
    conditions = {"clean": None, **PERTURBATIONS}
    for name, perturbation_fn in conditions.items():
        pooled_features, pooled_labels = [], []
        for source in COMBINED_SOURCES:
            set_seed(SEED)  # keep perturbations with randomness (RRC, color jitter) reproducible run-to-run
            features, labels = embed_indices(
                model, eval_bases[source], eval_indices_by_source[source],
                build_eval_transform(clip_preprocess, perturbation_fn), desc=f"{name}:{source}",
            )
            acc = evaluate_head(model, features, labels)
            results.append({"condition": name, "source": source, "n": len(eval_indices_by_source[source]),
                             "accuracy": round(acc, 4)})
            print(f"  {name:<18} {source:<10} accuracy: {acc:.4f}")
            pooled_features.append(features)
            pooled_labels.append(labels)

        pooled_acc = evaluate_head(model, torch.cat(pooled_features), torch.cat(pooled_labels))
        results.append({"condition": name, "source": "combined", "n": sum(f.size(0) for f in pooled_features),
                         "accuracy": round(pooled_acc, 4)})
        print(f"  {name:<18} {'combined':<10} accuracy: {pooled_acc:.4f}")

    table = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)

    print("\n=== Robustness table (per source + pooled, clean vs. each perturbation) ===")
    print(table.to_string(index=False))
    print(f"\nSaved to {TABLE_PATH}")


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

    if DATASET == "combined":
        run_combined(model, clip_preprocess)
        return

    # --- Step 1-2: datasets (transform set per-embedding-pass below) ---
    if DATASET not in DATASET_BUILDERS:
        raise ValueError(f"Unknown DATASET {DATASET!r}. Choose one of {list(DATASET_BUILDERS) + ['combined']}.")
    print(f"Dataset: {DATASET}")
    train_base, train_indices, test_base, eval_indices = DATASET_BUILDERS[DATASET]()

    # --- Step 4b: embed the training set ONCE (backbone-bound; everything
    # after this is just MLP training on the cached tensors) ---
    print("\nEmbedding training set (one backbone pass, cached for all epochs)...")
    train_features, train_labels = embed_indices(
        model, train_base, train_indices, build_train_transform(clip_preprocess), desc="train"
    )

    # --- Step 4: train only the head, on cached embeddings ---
    print("\nTraining head on cached embeddings...")
    train_head(model, train_features, train_labels, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY)

    # --- Checkpoint: save the trained head, then verify it reloads correctly ---
    save_head_checkpoint(model, HEAD_CHECKPOINT_PATH)
    reloaded_head, checkpoint_meta = load_head_checkpoint(HEAD_CHECKPOINT_PATH, device=DEVICE)
    model.head.eval()  # match reloaded_head's eval() mode — dropout must be off on both sides to compare outputs
    with torch.no_grad():
        probe = train_features[: min(8, train_features.size(0))]
        original_out = model.head(probe)
        reloaded_out = reloaded_head(probe)
    if not torch.allclose(original_out, reloaded_out):
        raise RuntimeError("Checkpoint round-trip mismatch: reloaded head does not match the trained head.")
    print(f"Checkpoint verified: reloaded head matches trained weights (meta={checkpoint_meta}).")

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
