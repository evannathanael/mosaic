import csv
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from src.data.clean_wildfake import clean_archives


def _png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_clean_archives_validates_and_deduplicates(tmp_path: Path):
    root = tmp_path / "raw"
    archive_path = root / "Images" / "Real"
    archive_path.mkdir(parents=True)
    payload = _png((10, 20, 30))
    _zip(archive_path / "afhq.zip", {
        "afhq/train/a.png": payload,
        "afhq/test/a_copy.png": payload,
        "afhq/train/b.txt": b"ignored",
        "afhq/train/bad.png": b"not an image",
    })
    report = clean_archives(root, tmp_path / "out", ["Images/Real/afhq.zip"], progress_every=0)
    assert report["valid_images"] == 2
    assert report["included_images"] == 1
    assert report["exact_duplicate_groups"] == 1
    assert report["rejected_images"] == 1
    with (tmp_path / "out" / "clean_manifest.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_holdout_archive_is_not_processed(tmp_path: Path):
    root = tmp_path / "raw"
    archive_path = root / "Images" / "Real"
    archive_path.mkdir(parents=True)
    _zip(archive_path / "coco.zip", {"coco/train/a.png": _png((1, 2, 3))})
    report = clean_archives(root, tmp_path / "out", ["Images/Real/coco.zip"], progress_every=0)
    assert report["valid_images"] == 0
    assert report["rejected_archives"] == 1
    assert "holdout archive" in (tmp_path / "out" / "rejected_archives.csv").read_text(encoding="utf-8")


def test_missing_archive_is_reported(tmp_path: Path):
    report = clean_archives(tmp_path / "raw", tmp_path / "out", ["Images/Diffusion_based/DDIM.zip"], progress_every=0)
    assert report["rejected_archives"] == 1
    report_json = json.loads((tmp_path / "out" / "cleaning_report.json").read_text(encoding="utf-8"))
    assert report_json["valid_images"] == 0
