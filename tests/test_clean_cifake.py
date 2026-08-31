import csv
import json

from PIL import Image

from src.data.clean_cifake import clean_cifake


def _save_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def test_clean_cifake_validates_and_handles_duplicate_leakage(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"

    _save_image(raw / "train" / "REAL" / "real.png", (10, 20, 30))
    _save_image(raw / "train" / "FAKE" / "fake.png", (100, 110, 120))
    _save_image(raw / "test" / "FAKE" / "duplicate.png", (100, 110, 120))
    bad = raw / "test" / "REAL" / "broken.png"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not an image")

    report = clean_cifake(raw, output)

    assert report["files_discovered"] == 4
    assert report["valid_images"] == 3
    assert report["included_images"] == 2
    assert report["rejected_images"] == 1
    assert report["cross_split_groups"] == 1

    with (output / "clean_manifest.csv").open(newline="", encoding="utf-8") as stream:
        clean_rows = list(csv.DictReader(stream))
    assert {row["relative_path"] for row in clean_rows} == {
        "train/REAL/real.png",
        "test/FAKE/duplicate.png",
    }

    saved_report = json.loads((output / "cleaning_report.json").read_text(encoding="utf-8"))
    assert saved_report == report


def test_clean_cifake_excludes_cross_label_duplicates(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    _save_image(raw / "train" / "REAL" / "same.png", (5, 6, 7))
    _save_image(raw / "train" / "FAKE" / "same.png", (5, 6, 7))

    report = clean_cifake(raw, output)

    assert report["cross_label_groups"] == 1
    assert report["included_images"] == 0
