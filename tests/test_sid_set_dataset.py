import csv

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from src.models.train_clip_head import SidSetParquetDataset, sid_set_label_policy


def _image_bytes(color):
    import io

    stream = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(stream, format="PNG")
    return stream.getvalue()


MANIFEST_FIELDS = [
    "parquet_file", "row_index", "img_id", "split", "source_label", "width", "height",
    "image_format", "image_mode", "image_sha256", "duplicate_sha256", "pixel_sha256",
    "dhash", "exact_duplicate_group", "include", "exclusion_reason",
]


def _write_shard(path, rows):
    """rows: list of (img_id, color, source_label)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "img_id": [r[0] for r in rows],
        "image": [{"bytes": _image_bytes(r[1]), "path": None} for r in rows],
        "width": [4 for _ in rows],
        "height": [4 for _ in rows],
        "label": [r[2] for r in rows],
    })
    pq.write_table(table, path)


def _write_manifest(path, rows):
    """rows: list of dicts with at least parquet_file/row_index/split/source_label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            full = {field: "" for field in MANIFEST_FIELDS}
            full.update(row)
            full["include"] = True
            writer.writerow(full)


def test_sid_set_label_policy():
    assert sid_set_label_policy(0) == 0
    assert sid_set_label_policy(1) == 1
    assert sid_set_label_policy(2) is None


def test_sid_set_dataset_excludes_tampered_and_respects_split(tmp_path):
    raw_root = tmp_path / "raw"
    _write_shard(raw_root / "data" / "train-0.parquet", [
        ("real_0", (10, 20, 30), 0),
        ("fake_0", (40, 50, 60), 1),
        ("tampered_0", (70, 80, 90), 2),
    ])
    _write_shard(raw_root / "data" / "validation-0.parquet", [
        ("real_1", (11, 21, 31), 0),
    ])

    manifest_path = tmp_path / "processed" / "clean_manifest.csv"
    _write_manifest(manifest_path, [
        {"parquet_file": "data/train-0.parquet", "row_index": 0, "img_id": "real_0", "split": "train", "source_label": 0},
        {"parquet_file": "data/train-0.parquet", "row_index": 1, "img_id": "fake_0", "split": "train", "source_label": 1},
        {"parquet_file": "data/train-0.parquet", "row_index": 2, "img_id": "tampered_0", "split": "train", "source_label": 2},
        {"parquet_file": "data/validation-0.parquet", "row_index": 0, "img_id": "real_1", "split": "validation", "source_label": 0},
    ])

    train_ds = SidSetParquetDataset(manifest_path, raw_root, split="train")
    assert len(train_ds) == 2  # tampered row excluded
    labels = sorted(label for _, label in train_ds.samples)
    assert labels == [0, 1]

    val_ds = SidSetParquetDataset(manifest_path, raw_root, split="validation")
    assert len(val_ds) == 1

    image, label = train_ds[0]
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert label in (0, 1)


def test_sid_set_dataset_applies_transform(tmp_path):
    raw_root = tmp_path / "raw"
    _write_shard(raw_root / "data" / "train-0.parquet", [("real_0", (10, 20, 30), 0)])
    manifest_path = tmp_path / "processed" / "clean_manifest.csv"
    _write_manifest(manifest_path, [
        {"parquet_file": "data/train-0.parquet", "row_index": 0, "img_id": "real_0", "split": "train", "source_label": 0},
    ])

    calls = []

    def fake_transform(image):
        calls.append(image.size)
        return "transformed"

    ds = SidSetParquetDataset(manifest_path, raw_root, split="train", transform=fake_transform)
    image, label = ds[0]
    assert image == "transformed"
    assert label == 0
    assert calls == [(4, 4)]


def test_sid_set_dataset_raises_on_empty_split(tmp_path):
    raw_root = tmp_path / "raw"
    _write_shard(raw_root / "data" / "train-0.parquet", [("tampered_only", (1, 2, 3), 2)])
    manifest_path = tmp_path / "processed" / "clean_manifest.csv"
    _write_manifest(manifest_path, [
        {"parquet_file": "data/train-0.parquet", "row_index": 0, "img_id": "tampered_only", "split": "train", "source_label": 2},
    ])

    try:
        SidSetParquetDataset(manifest_path, raw_root, split="train")
        assert False, "expected ValueError for an all-tampered split"
    except ValueError:
        pass
