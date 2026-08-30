"""Compatibility layer so src/eval/ scripts can consume whatever checkpoint
and raw-data shape actually exists on disk, without forking each of the four
eval scripts into per-format branches.

Three mismatches this bridges:
  1. Checkpoint format — src/models/train.py (classifier.py's AIGCDetector)
     saves {"state_dict", "temperature"}; src/models/train_clip_head.py
     (clip_aigc_head.py's AIGCClipDetector) saves a structurally different
     {"head_state_dict", "embed_dim", "hidden_dim", "dropout",
     "clip_model_name", "clip_pretrained"}. load_eval_model() detects which
     one a checkpoint is and returns a common adapter interface either way:
     .eval(), .predict_proba(images), .embed(images),
     .logits_from_embedding(embedding), .temperature, .save(path).
  2. Raw data layout — scan_dataset() (src/data/dataset.py) expects a flat
     data/raw/<name>/{real,ai}/ layout. train_clip_head.py instead reads
     CIFAKE straight from Kaggle's native data/raw/CIFAKE/{train,test}/
     {REAL,FAKE} shape. scan_cifake_nested() reads that same native shape
     and returns ordinary Sample objects.
  3. Non-file-backed sources — SID-Set images live embedded in Parquet shards
     (referenced by (parquet_file, row_index)); WildFake images live inside
     ZIP archives (referenced by (archive_path, member_path)). AIGCDataset
     only knows `Image.open(sample.path)`, so it can't read either. Rather
     than touch AIGCDataset (src/data/dataset.py, Vicky's), samples from
     these sources get a pseudo-path scheme string
     ("sidset-parquet://<parquet_file>#<row_index>" /
     "wildfake-zip://<archive_path>#<member_path>") and EvalDataset (this
     file) dispatches on that scheme instead of assuming a plain file path.
     scan_all_sources() combines CIFAKE + SID-Set + WildFake into one Sample
     pool, skipping (with a warning) whichever sources aren't on disk yet.
"""
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset import Sample
from src.data.transforms import named_eval_transform
from src.models.classifier import AIGCDetector
from src.models.clip_aigc_head import AIGCClipDetector
from src.models.train_clip_head import sid_set_label_policy
from src.utils import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

SID_SET_SCHEME = "sidset-parquet://"
WILDFAKE_SCHEME = "wildfake-zip://"


def scan_cifake_nested(cifake_root: Path) -> list[Sample]:
    """Reads data/raw/CIFAKE/{train,test}/{REAL,FAKE}/... (Kaggle's native
    layout, as train_clip_head.py's DATA_ROOT expects) and returns a flat
    Sample list — train/test are merged into one pool since split_samples()
    does its own re-split anyway; only the REAL/FAKE label matters here.
    """
    cifake_root = Path(cifake_root)
    label_map = {"REAL": 0, "FAKE": 1}
    samples: list[Sample] = []
    for split_dir in ("train", "test"):
        split_path = cifake_root / split_dir
        if not split_path.exists():
            continue
        for label_name, label in label_map.items():
            label_dir = split_path / label_name
            if not label_dir.exists():
                continue
            for img_path in label_dir.rglob("*"):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                samples.append(Sample(path=str(img_path), label=label, generator="cifake"))
    return samples


def scan_sid_set_manifest(manifest_path: Path) -> list[Sample]:
    """Reads Vicky's cleaned SID-Set manifest (data/processed/sid_set/
    clean_manifest.csv) and returns Sample objects referencing Parquet rows
    via a pseudo-path (see module docstring) — actual image bytes are never
    touched here, only metadata. Applies the same binary label policy
    train_clip_head.py's SidSetParquetDataset uses (0/1 kept, 2=tampered
    dropped), reused from there rather than redefined, plus an `include`
    check that SidSetParquetDataset itself does NOT do (it only filters by
    split) — clean_manifest.csv marks duplicate/rejected rows include=False,
    so respecting it here is stricter than the existing training loader.
    """
    manifest_path = Path(manifest_path)
    samples: list[Sample] = []
    with open(manifest_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("include") != "True":
                continue
            binary_label = sid_set_label_policy(int(row["source_label"]))
            if binary_label is None:
                continue
            path = f"{SID_SET_SCHEME}{row['parquet_file']}#{row['row_index']}"
            samples.append(Sample(path=path, label=binary_label, generator="sid_set"))
    return samples


def scan_wildfake_manifest(manifest_path: Path) -> list[Sample]:
    """Reads Vicky's cleaned WildFake manifest (data/processed/wildfake/
    clean_manifest.csv) and returns Sample objects referencing ZIP members
    via a pseudo-path (see module docstring). generator is tagged
    "wildfake_<generator>" (e.g. "wildfake_afhq", "wildfake_DDIM"), matching
    scan_dataset()'s existing wildfake generator-tagging convention in
    src/data/dataset.py, so per-source breakdowns line up either way.
    """
    manifest_path = Path(manifest_path)
    samples: list[Sample] = []
    with open(manifest_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("include") != "True":
                continue
            path = f"{WILDFAKE_SCHEME}{row['archive_path']}#{row['member_path']}"
            samples.append(
                Sample(path=path, label=int(row["source_label"]), generator=f"wildfake_{row['generator']}")
            )
    return samples


def scan_all_sources(config: dict) -> list[Sample]:
    """Combines CIFAKE + SID-Set + WildFake into one Sample pool, skipping
    (with a warning, not an error) whichever source isn't actually on disk —
    mirrors how robustness.py already tolerates an empty split rather than
    crashing when a dataset hasn't been downloaded yet.
    """
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    samples: list[Sample] = []

    cifake_root = raw_dir / "CIFAKE"
    if cifake_root.exists():
        cifake_samples = scan_cifake_nested(cifake_root)
        logger.info("CIFAKE: %d samples", len(cifake_samples))
        samples.extend(cifake_samples)
    else:
        logger.warning("CIFAKE not found at %s — skipping", cifake_root)

    sid_set_manifest = processed_dir / "sid_set" / "clean_manifest.csv"
    if sid_set_manifest.exists():
        sid_set_samples = scan_sid_set_manifest(sid_set_manifest)
        logger.info("SID-Set: %d samples", len(sid_set_samples))
        samples.extend(sid_set_samples)
    else:
        logger.warning("SID-Set manifest not found at %s — skipping", sid_set_manifest)

    wildfake_manifest = processed_dir / "wildfake" / "clean_manifest.csv"
    if wildfake_manifest.exists():
        wildfake_samples = scan_wildfake_manifest(wildfake_manifest)
        logger.info("WildFake: %d samples", len(wildfake_samples))
        samples.extend(wildfake_samples)
    else:
        logger.warning("WildFake manifest not found at %s — skipping", wildfake_manifest)

    return samples


class EvalDataset(Dataset):
    """Drop-in replacement for AIGCDataset (src/data/dataset.py) that also
    understands the pseudo-path schemes scan_sid_set_manifest() and
    scan_wildfake_manifest() produce, in addition to plain file paths.
    Returns the exact same dict shape AIGCDataset does, so nothing downstream
    in the four eval scripts needs to change. Eval-only (mode="eval") — none
    of the eval scripts use AIGCDataset's mode="train" path, so it's not
    reproduced here.
    """

    def __init__(
        self,
        samples: list[Sample],
        config: dict,
        eval_transform_name: str = "clean",
        sid_set_raw_root: Path | None = None,
        wildfake_raw_root: Path | None = None,
    ):
        self.samples = samples
        self.image_size = config["data"]["image_size"]
        self.eval_transform_name = eval_transform_name
        raw_dir = Path(config["data"]["raw_dir"])
        self.sid_set_raw_root = Path(sid_set_raw_root) if sid_set_raw_root else raw_dir / "sid_set"
        self.wildfake_raw_root = Path(wildfake_raw_root) if wildfake_raw_root else raw_dir / "wildfake"
        self._parquet_cache: dict[str, list] = {}
        self._zip_cache: dict[str, zipfile.ZipFile] = {}

    def __len__(self):
        return len(self.samples)

    def _read_sid_set_image(self, path: str) -> Image.Image:
        parquet_file, row_index = path[len(SID_SET_SCHEME):].split("#")
        if parquet_file not in self._parquet_cache:
            table = pq.read_table(self.sid_set_raw_root / parquet_file, columns=["image"])
            self._parquet_cache[parquet_file] = table.column("image").to_pylist()
        image_value = self._parquet_cache[parquet_file][int(row_index)]
        raw = image_value.get("bytes") if isinstance(image_value, dict) else bytes(image_value)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _read_wildfake_image(self, path: str) -> Image.Image:
        archive_path, member_path = path[len(WILDFAKE_SCHEME):].split("#")
        if archive_path not in self._zip_cache:
            self._zip_cache[archive_path] = zipfile.ZipFile(self.wildfake_raw_root / archive_path)
        raw = self._zip_cache[archive_path].read(member_path)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if sample.path.startswith(SID_SET_SCHEME):
            pil_image = self._read_sid_set_image(sample.path)
        elif sample.path.startswith(WILDFAKE_SCHEME):
            pil_image = self._read_wildfake_image(sample.path)
        else:
            pil_image = Image.open(sample.path).convert("RGB")

        img = named_eval_transform(self.eval_transform_name, np.array(pil_image))
        img = Image.fromarray(img).resize((self.image_size, self.image_size))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5

        return {
            "image": img_array.transpose(2, 0, 1),
            "label": sample.label,
            "path": sample.path,
            "generator": sample.generator,
        }


class _StandardAdapter:
    """Wraps AIGCDetector (classifier.py) — the config-driven SharedBackbone
    + ClassifierHead + temperature model that src/models/train.py produces.
    """

    kind = "standard"

    def __init__(self, model: AIGCDetector):
        self.model = model
        self.temperature = model.temperature  # same nn.Parameter object — in-place edits stay in sync

    def eval(self):
        self.model.eval()

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.predict_proba(images)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.backbone(images)

    def logits_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.model.classifier(embedding)

    def save(self, path: str) -> None:
        self.model.save(path)


class _ClipHeadAdapter:
    """Wraps AIGCClipDetector (clip_aigc_head.py) — the frozen-CLIP + MlpHead
    model that src/models/train_clip_head.py produces. That checkpoint format
    has no temperature field of its own; one is added here (defaulting to 1.0)
    so calibration.py can fit/persist one without needing train_clip_head.py
    to change.
    """

    kind = "clip_head"

    def __init__(self, model: AIGCClipDetector, meta: dict, device: str):
        self.model = model
        self.meta = meta
        self.temperature = torch.nn.Parameter(
            torch.tensor([meta.get("temperature", 1.0)], device=device), requires_grad=False
        )

    def eval(self):
        self.model.eval()

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images)

    def logits_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.model.head(embedding)

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.embed(images)
            logits = self.logits_from_embedding(embedding) / self.temperature
            return torch.sigmoid(logits)

    def save(self, path: str) -> None:
        torch.save(
            {
                "head_state_dict": self.model.head.state_dict(),
                "embed_dim": self.model.embed_dim,
                "hidden_dim": self.meta["hidden_dim"],
                "dropout": self.meta["dropout"],
                "clip_model_name": self.meta["clip_model_name"],
                "clip_pretrained": self.meta["clip_pretrained"],
                "temperature": self.temperature.item(),
            },
            path,
        )


def load_eval_model(checkpoint_path: str, config: dict, device: str = "cpu"):
    """Detects which of the two checkpoint formats `checkpoint_path` is and
    returns the matching adapter (_StandardAdapter or _ClipHeadAdapter),
    both exposing the same interface the eval scripts rely on.
    """
    raw = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in raw:
        model = AIGCDetector.load(checkpoint_path, config, device=device)
        return _StandardAdapter(model)

    if "head_state_dict" in raw:
        model = AIGCClipDetector(
            clip_model_name=raw["clip_model_name"],
            clip_pretrained=raw["clip_pretrained"],
            hidden_dim=raw["hidden_dim"],
            dropout=raw["dropout"],
        ).to(device)
        model.head.load_state_dict(raw["head_state_dict"])
        model.eval()
        meta = {
            "hidden_dim": raw["hidden_dim"],
            "dropout": raw["dropout"],
            "clip_model_name": raw["clip_model_name"],
            "clip_pretrained": raw["clip_pretrained"],
            "temperature": raw.get("temperature", 1.0),
        }
        return _ClipHeadAdapter(model, meta, device)

    raise ValueError(
        f"Unrecognized checkpoint format at {checkpoint_path}: keys={sorted(raw.keys())}. "
        "Expected either 'state_dict' (classifier.py's AIGCDetector) or "
        "'head_state_dict' (clip_aigc_head.py's AIGCClipDetector)."
    )
