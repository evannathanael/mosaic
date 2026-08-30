import csv
import io
import zipfile

from PIL import Image

from src.models.train_clip_head import WildfakeArchiveDataset, wildfake_label_policy


def _png(color):
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(stream, format="PNG")
    return stream.getvalue()


def _zip(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


MANIFEST_FIELDS = [
    "archive_path", "member_path", "source", "generator", "split", "source_label", "is_advanced",
    "width", "height", "image_format", "image_mode", "image_sha256", "pixel_sha256", "dhash",
    "exact_duplicate_group", "include", "exclusion_reason",
]


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            full = {field: "" for field in MANIFEST_FIELDS}
            full.update(row)
            full["include"] = True
            writer.writerow(full)


def test_wildfake_label_policy():
    assert wildfake_label_policy(0) == 0
    assert wildfake_label_policy(1) == 1
    assert wildfake_label_policy(2) is None


def test_wildfake_dataset_reads_zip_members_and_respects_split(tmp_path):
    raw_root = tmp_path / "raw"
    _zip(raw_root / "Images" / "Real" / "afhq.zip", {
        "afhq/train/real_0.png": _png((10, 20, 30)),
    })
    _zip(raw_root / "Images" / "Diffusion_based" / "DDIM.zip", {
        "ddim/train/fake_0.png": _png((40, 50, 60)),
    })
    _zip(raw_root / "Images" / "Real" / "afhq_val.zip", {
        "afhq/validation/real_1.png": _png((11, 21, 31)),
    })

    manifest_path = tmp_path / "processed" / "clean_manifest.csv"
    _write_manifest(manifest_path, [
        {"archive_path": "Images/Real/afhq.zip", "member_path": "afhq/train/real_0.png",
         "split": "train", "source_label": 0},
        {"archive_path": "Images/Diffusion_based/DDIM.zip", "member_path": "ddim/train/fake_0.png",
         "split": "train", "source_label": 1},
        {"archive_path": "Images/Real/afhq_val.zip", "member_path": "afhq/validation/real_1.png",
         "split": "validation", "source_label": 0},
    ])

    train_ds = WildfakeArchiveDataset(manifest_path, raw_root, split="train")
    assert len(train_ds) == 2
    labels = sorted(label for _, label in train_ds.samples)
    assert labels == [0, 1]

    val_ds = WildfakeArchiveDataset(manifest_path, raw_root, split="validation")
    assert len(val_ds) == 1

    image, label = train_ds[0]
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert label in (0, 1)


def test_wildfake_dataset_applies_transform(tmp_path):
    raw_root = tmp_path / "raw"
    _zip(raw_root / "Images" / "Real" / "afhq.zip", {"afhq/train/real_0.png": _png((10, 20, 30))})
    manifest_path = tmp_path / "processed" / "clean_manifest.csv"
    _write_manifest(manifest_path, [
        {"archive_path": "Images/Real/afhq.zip", "member_path": "afhq/train/real_0.png",
         "split": "train", "source_label": 0},
    ])

    calls = []

    def fake_transform(image):
        calls.append(image.size)
        return "transformed"

    ds = WildfakeArchiveDataset(manifest_path, raw_root, split="train", transform=fake_transform)
    image, label = ds[0]
    assert image == "transformed"
    assert label == 0
    assert calls == [(4, 4)]


def test_wildfake_dataset_raises_on_empty_split(tmp_path):
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "processed" / "clean_manifest.csv"
    _write_manifest(manifest_path, [
        {"archive_path": "Images/Real/afhq.zip", "member_path": "afhq/train/real_0.png",
         "split": "train", "source_label": 0},
    ])

    try:
        WildfakeArchiveDataset(manifest_path, raw_root, split="validation")
        assert False, "expected ValueError when the requested split has no rows"
    except ValueError:
        pass
