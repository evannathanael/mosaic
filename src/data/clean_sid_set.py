"""Validate SID-Set Parquet shards and build a leakage-aware manifest.

SID-Set stores the image inside each Parquet row. This cleaner intentionally
keeps the Parquet files immutable and records ``parquet_file`` + ``row_index``
so a downstream loader can read image bytes on demand.

Usage::

    python -m src.data.clean_sid_set
    python -m src.data.clean_sid_set --input data/raw/sid_set_hf --output data/processed/sid_set

The original SID labels are preserved as ``source_label``:

    0 = real, 1 = full synthetic, 2 = tampered

No binary training-label policy is applied here.
"""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import io
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps, UnidentifiedImageError


REQUIRED_COLUMNS = {"img_id", "image", "width", "height", "label"}
EXPECTED_LABELS = {0, 1, 2}


@dataclass
class ImageRecord:
    parquet_file: str
    row_index: int
    img_id: str
    split: str
    source_label: int
    width: int
    height: int
    image_format: str
    image_mode: str
    image_sha256: str
    duplicate_sha256: str
    pixel_sha256: str
    dhash: str
    exact_duplicate_group: str = ""
    include: bool = True
    exclusion_reason: str = ""


def _pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _dhash(image: Image.Image, hash_size: int = 8) -> str:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    gray = image.convert("L").resize((hash_size + 1, hash_size), resampling)
    get_pixels = getattr(gray, "get_flattened_data", gray.getdata)
    pixels = list(get_pixels())
    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            bits = (bits << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return f"{bits:0{hash_size * hash_size // 4}x}"


def _image_bytes(value) -> bytes:
    """Extract bytes from a Hugging Face image struct scalar."""
    item = value.as_py() if hasattr(value, "as_py") else value
    if isinstance(item, dict):
        raw = item.get("bytes")
        if raw is not None:
            return raw
        raise ValueError("image.bytes is missing (row contains no embedded image)")
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    raise ValueError("image column is not an embedded byte payload")


def _decode_image(raw: bytes, visual_hashes: bool = False):
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            image_format = probe.format or "unknown"
            image_mode = probe.mode
            width, height = probe.size
            probe.verify()
        rgb = None
        if visual_hashes:
            with Image.open(io.BytesIO(raw)) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                rgb.load()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"unreadable embedded image: {exc}") from exc
    if width < 1 or height < 1:
        raise ValueError("image has an invalid size")
    return width, height, image_format, image_mode, (rgb if visual_hashes else None)


def _split_from_path(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("train-"):
        return "train"
    if name.startswith("validation-"):
        return "validation"
    raise ValueError("Parquet filename must start with train- or validation-")


def _read_shard(
    path: Path,
    root: Path,
    progress_state: dict,
    visual_hashes: bool = False,
) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    split = _split_from_path(path)
    relative = path.relative_to(root).as_posix()
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    missing = REQUIRED_COLUMNS - available
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")

    records: list[ImageRecord] = []
    rejected: list[dict[str, str]] = []
    row_index = 0
    for batch in parquet.iter_batches(
        columns=["img_id", "image", "width", "height", "label"],
        batch_size=128,
    ):
        img_ids = batch.column("img_id")
        images = batch.column("image")
        widths = batch.column("width")
        heights = batch.column("height")
        labels = batch.column("label")
        for offset in range(batch.num_rows):
            current_index = row_index + offset
            try:
                source_label = labels[offset].as_py()
                if source_label not in EXPECTED_LABELS:
                    raise ValueError(f"unsupported source label: {source_label!r}")
                width = widths[offset].as_py()
                height = heights[offset].as_py()
                if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
                    raise ValueError("invalid metadata dimensions")
                raw = _image_bytes(images[offset])
                decoded_width, decoded_height, image_format, image_mode, image = _decode_image(raw, visual_hashes)
                if (decoded_width, decoded_height) != (width, height):
                    raise ValueError(
                        f"metadata dimensions {width}x{height} do not match decoded "
                        f"image {decoded_width}x{decoded_height}"
                    )
                records.append(ImageRecord(
                    parquet_file=relative,
                    row_index=current_index,
                    img_id=str(img_ids[offset].as_py()),
                    split=split,
                    source_label=source_label,
                    width=width,
                    height=height,
                    image_format=image_format,
                    image_mode=image_mode,
                    image_sha256=hashlib.sha256(raw).hexdigest(),
                    duplicate_sha256=hashlib.sha256(raw).hexdigest(),
                    pixel_sha256=_pixel_sha256(image) if visual_hashes else "",
                    dhash=_dhash(image) if visual_hashes else "",
                ))
            except (ValueError, TypeError, OverflowError) as exc:
                rejected.append({
                    "parquet_file": relative,
                    "row_index": str(current_index),
                    "reason": str(exc),
                })
        row_index += batch.num_rows
        progress_state["rows"] += batch.num_rows
        if progress_state["progress_every"] > 0 and (
            progress_state["rows"] % progress_state["progress_every"] < batch.num_rows
        ):
            print(
                f"  scanned {progress_state['rows']:,} rows "
                f"(valid {progress_state['valid'] + len(records):,}, "
                f"rejected {progress_state['rejected'] + len(rejected):,})",
                file=sys.stderr,
                flush=True,
            )
    return records, rejected


def _apply_exact_duplicate_policy(records: list[ImageRecord]) -> dict[str, int]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.duplicate_sha256].append(record)

    stats = Counter()
    duplicate_index = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        duplicate_index += 1
        group_id = f"exact_{duplicate_index:06d}"
        for member in members:
            member.exact_duplicate_group = group_id
        stats["exact_duplicate_groups"] += 1
        stats["exact_duplicate_images"] += len(members)

        labels = {member.source_label for member in members}
        splits = {member.split for member in members}
        if len(labels) > 1:
            stats["cross_label_groups"] += 1
            for member in members:
                member.include = False
                member.exclusion_reason = "exact_duplicate_label_conflict"
            continue

        if len(splits) > 1:
            stats["cross_split_groups"] += 1

        # Keep the official validation copy when a duplicate crosses splits.
        ordered = sorted(
            members,
            key=lambda item: (item.split != "validation", item.parquet_file, item.row_index),
        )
        for duplicate in ordered[1:]:
            duplicate.include = False
            duplicate.exclusion_reason = "exact_duplicate"
    return dict(stats)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_sid_set(
    input_dir: str | Path,
    output_dir: str | Path,
    progress_every: int = 0,
    visual_hashes: bool = False,
    workers: int = 1,
) -> dict:
    root = Path(input_dir).resolve()
    output = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SID-Set input directory does not exist: {root}")

    shards = sorted(root.rglob("*.parquet"))
    if not shards:
        raise ValueError(f"No Parquet shards found under {root}")
    if progress_every > 0:
        print(f"Found {len(shards):,} Parquet shards. Validating embedded images...", file=sys.stderr, flush=True)

    if workers < 1:
        raise ValueError("workers must be at least 1")
    workers = min(workers, len(shards))
    output.mkdir(parents=True, exist_ok=True)
    temp_db = output / ".sid_cleaning_tmp.sqlite3"
    if temp_db.exists():
        temp_db.unlink()

    connection = sqlite3.connect(temp_db)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute(
        """CREATE TABLE records (
            id INTEGER PRIMARY KEY,
            parquet_file TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            img_id TEXT NOT NULL,
            split TEXT NOT NULL,
            source_label INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            image_format TEXT NOT NULL,
            image_mode TEXT NOT NULL,
            image_sha256 TEXT NOT NULL,
            duplicate_sha256 TEXT NOT NULL,
            pixel_sha256 TEXT NOT NULL,
            dhash TEXT NOT NULL,
            exact_duplicate_group TEXT NOT NULL DEFAULT '',
            include INTEGER NOT NULL DEFAULT 1,
            exclusion_reason TEXT NOT NULL DEFAULT ''
        )"""
    )
    connection.execute("CREATE INDEX records_duplicate_idx ON records(duplicate_sha256)")
    connection.execute(
        "CREATE TABLE rejected (parquet_file TEXT, row_index INTEGER, reason TEXT)"
    )
    connection.commit()

    def process_shard(shard: Path):
        local_state = {"rows": 0, "valid": 0, "rejected": 0, "progress_every": 0}
        try:
            shard_records, shard_rejected = _read_shard(
                shard, root, local_state, visual_hashes=visual_hashes
            )
            return shard, shard_records, shard_rejected, None
        except Exception as exc:
            return shard, [], [], str(exc)

    def store_result(result):
        shard, shard_records, shard_rejected, shard_error = result
        if shard_error is not None:
            rejected_shards.append({
                "parquet_file": shard.relative_to(root).as_posix(),
                "reason": shard_error,
            })
        connection.executemany(
            """INSERT INTO records (
                parquet_file, row_index, img_id, split, source_label, width, height,
                image_format, image_mode, image_sha256, duplicate_sha256,
                pixel_sha256, dhash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(
                record.parquet_file, record.row_index, record.img_id, record.split,
                record.source_label, record.width, record.height, record.image_format,
                record.image_mode, record.image_sha256, record.duplicate_sha256,
                record.pixel_sha256, record.dhash,
            ) for record in shard_records],
        )
        connection.executemany(
            "INSERT INTO rejected (parquet_file, row_index, reason) VALUES (?, ?, ?)",
            [(
                item["parquet_file"], int(item["row_index"]), item["reason"]
            ) for item in shard_rejected],
        )
        connection.commit()
        return shard, len(shard_records), len(shard_rejected), shard_error

    rejected_shards: list[dict[str, str]] = []
    completed = 0
    valid_count = 0
    rejected_count = 0
    shard_iterator = iter(shards)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        for _ in range(workers):
            try:
                shard = next(shard_iterator)
            except StopIteration:
                break
            pending[executor.submit(process_shard, shard)] = shard
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                pending.pop(future)
                shard, shard_valid, shard_rejected, shard_error = store_result(future.result())
                completed += 1
                valid_count += shard_valid
                rejected_count += shard_rejected
                if progress_every > 0:
                    detail = (
                        f"{shard_valid:,} valid, {shard_rejected:,} rejected"
                        if shard_error is None else f"shard error: {shard_error}"
                    )
                    print(
                        f"  completed shard {completed:,}/{len(shards):,}: {shard.name} ({detail})",
                        file=sys.stderr,
                        flush=True,
                    )
                try:
                    next_shard = next(shard_iterator)
                except StopIteration:
                    continue
                pending[executor.submit(process_shard, next_shard)] = next_shard

    # Assign stable duplicate groups using SQL rather than retaining every row
    # in Python memory.
    duplicate_stats = Counter()
    duplicate_index = 0
    duplicate_hashes = connection.execute(
        "SELECT duplicate_sha256 FROM records GROUP BY duplicate_sha256 HAVING COUNT(*) > 1 ORDER BY duplicate_sha256"
    ).fetchall()
    for (duplicate_hash,) in duplicate_hashes:
        members = connection.execute(
            """SELECT id, split, source_label FROM records
               WHERE duplicate_sha256 = ?
               ORDER BY (split = 'validation') DESC, parquet_file, row_index""",
            (duplicate_hash,),
        ).fetchall()
        duplicate_index += 1
        group_id = f"exact_{duplicate_index:06d}"
        connection.executemany(
            "UPDATE records SET exact_duplicate_group = ? WHERE id = ?",
            [(group_id, member[0]) for member in members],
        )
        duplicate_stats["exact_duplicate_groups"] += 1
        duplicate_stats["exact_duplicate_images"] += len(members)
        labels = {member[2] for member in members}
        splits = {member[1] for member in members}
        if len(labels) > 1:
            duplicate_stats["cross_label_groups"] += 1
            connection.executemany(
                "UPDATE records SET include = 0, exclusion_reason = ? WHERE id = ?",
                [("exact_duplicate_label_conflict", member[0]) for member in members],
            )
        else:
            if len(splits) > 1:
                duplicate_stats["cross_split_groups"] += 1
            connection.executemany(
                "UPDATE records SET include = 0, exclusion_reason = ? WHERE id = ?",
                [("exact_duplicate", member[0]) for member in members[1:]],
            )
    connection.commit()

    fields = list(ImageRecord.__dataclass_fields__)
    select_fields = ", ".join(fields)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream, \
            (output / "clean_manifest.csv").open("w", newline="", encoding="utf-8") as clean_stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        clean_writer = csv.DictWriter(clean_stream, fieldnames=fields)
        writer.writeheader()
        clean_writer.writeheader()
        cursor = connection.execute(f"SELECT {select_fields} FROM records ORDER BY parquet_file, row_index")
        for row in cursor:
            row_dict = dict(zip(fields, row))
            row_dict["include"] = bool(row_dict["include"])
            writer.writerow(row_dict)
            if row_dict["include"]:
                clean_writer.writerow(row_dict)

    with (output / "rejected.csv").open("w", newline="", encoding="utf-8") as stream:
        rejected_writer = csv.DictWriter(stream, fieldnames=["parquet_file", "row_index", "reason"])
        rejected_writer.writeheader()
        for row in connection.execute("SELECT parquet_file, row_index, reason FROM rejected ORDER BY parquet_file, row_index"):
            rejected_writer.writerow(dict(zip(["parquet_file", "row_index", "reason"], row)))
    _write_csv(output / "rejected_shards.csv", rejected_shards, ["parquet_file", "reason"])

    review_fields = ["review_group", "dhash", "parquet_file", "row_index", "img_id", "split", "source_label"]
    review_group_count = 0
    with (output / "near_duplicate_review.csv").open("w", newline="", encoding="utf-8") as stream:
        review_writer = csv.DictWriter(stream, fieldnames=review_fields)
        review_writer.writeheader()
        dhash_groups = connection.execute(
            "SELECT dhash FROM records WHERE dhash != '' GROUP BY dhash HAVING COUNT(*) > 1 ORDER BY dhash"
        ).fetchall()
        for (dhash,) in dhash_groups:
            members = connection.execute(
                "SELECT pixel_sha256, parquet_file, row_index, img_id, split, source_label FROM records WHERE dhash = ? ORDER BY parquet_file, row_index",
                (dhash,),
            ).fetchall()
            if len({member[0] for member in members}) == 1:
                continue
            review_group_count += 1
            for member in members:
                review_writer.writerow({
                    "review_group": f"review_{review_group_count:06d}",
                    "dhash": dhash,
                    "parquet_file": member[1],
                    "row_index": member[2],
                    "img_id": member[3],
                    "split": member[4],
                    "source_label": member[5],
                })

    included_count = connection.execute("SELECT COUNT(*) FROM records WHERE include = 1").fetchone()[0]
    dimensions = {
        f"{width}x{height}": count
        for width, height, count in connection.execute(
            "SELECT width, height, COUNT(*) FROM records GROUP BY width, height ORDER BY width, height"
        )
    }
    counts = {
        f"{split}/label_{label}": count
        for split, label, count in connection.execute(
            "SELECT split, source_label, COUNT(*) FROM records WHERE include = 1 GROUP BY split, source_label ORDER BY split, source_label"
        )
    }
    report = {
        "input_dir": str(root),
        "output_dir": str(output),
        "shards_discovered": len(shards),
        "rows_scanned": valid_count + rejected_count,
        "valid_rows": valid_count,
        "included_rows": included_count,
        "rejected_rows": rejected_count,
        "rejected_shards": len(rejected_shards),
        "counts_by_split_and_source_label": counts,
        "dimensions": dimensions,
        "near_duplicate_review_groups": review_group_count,
        "visual_hashes_enabled": visual_hashes,
        "workers": workers,
        **dict(duplicate_stats),
    }
    with (output / "cleaning_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    connection.close()
    temp_db.unlink(missing_ok=True)
    if progress_every > 0:
        print(f"Done. Reports written to {output}", file=sys.stderr, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SID-Set Parquet shards and build a clean manifest.")
    parser.add_argument("--input", default="data/raw/sid_set_hf", help="Raw SID-Set directory.")
    parser.add_argument("--output", default="data/processed/sid_set", help="Manifest output directory.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N rows (0 disables progress output).",
    )
    parser.add_argument(
        "--visual-hashes",
        action="store_true",
        help="Also compute canonical pixel/dHash values (slower; for visual duplicate review).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of Parquet shards to validate concurrently (default: up to 4).",
    )
    args = parser.parse_args()
    report = clean_sid_set(
        args.input,
        args.output,
        progress_every=args.progress_every,
        visual_hashes=args.visual_hashes,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
