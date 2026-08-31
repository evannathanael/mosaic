import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from src.data.clean_sid_set import clean_sid_set


def _image_bytes(color):
    import io

    stream = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(stream, format="PNG")
    return stream.getvalue()


def _write_shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "img_id": [row[0] for row in rows],
        "image": [{"bytes": row[1], "path": None} for row in rows],
        "mask": [{"bytes": None, "path": None} for _ in rows],
        "width": [4 for _ in rows],
        "height": [4 for _ in rows],
        "label": [row[2] for row in rows],
    })
    pq.write_table(table, path)


def test_clean_sid_preserves_source_labels_and_excludes_cross_split_duplicates(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    image = _image_bytes((10, 20, 30))
    _write_shard(raw / "data" / "train-00000-of-00001.parquet", [
        ("real", image, 0),
        ("fake", _image_bytes((100, 110, 120)), 1),
    ])
    _write_shard(raw / "data" / "validation-00000-of-00001.parquet", [
        ("real-copy", image, 0),
    ])

    report = clean_sid_set(raw, output)

    assert report["rows_scanned"] == 3
    assert report["valid_rows"] == 3
    assert report["included_rows"] == 2
    assert report["cross_split_groups"] == 1

    with (output / "clean_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["source_label"] for row in rows} == {"0", "1"}
    assert {row["split"] for row in rows} == {"train", "validation"}

    saved = json.loads((output / "cleaning_report.json").read_text(encoding="utf-8"))
    assert saved == report


def test_clean_sid_rejects_bad_bytes_and_labels(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_shard(raw / "data" / "train-00000-of-00001.parquet", [
        ("bad-bytes", b"not an image", 0),
        ("bad-label", _image_bytes((1, 2, 3)), 9),
    ])

    report = clean_sid_set(raw, output)

    assert report["rows_scanned"] == 2
    assert report["valid_rows"] == 0
    assert report["rejected_rows"] == 2
