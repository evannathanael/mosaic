"""Validate CIFAKE and build leakage-aware manifests.

The raw dataset is never modified. Valid images are inventoried in CSV files,
exact duplicates are handled deterministically, and perceptual-hash matches are
reported for manual review rather than removed automatically.

Expected CIFAKE layout (directory names are case-insensitive)::

    data/raw/cifake/
        train/{REAL,FAKE}/*
        test/{REAL,FAKE}/*

Usage::

    python -m src.data.clean_cifake
    python -m src.data.clean_cifake --input path/to/cifake --output data/processed/cifake
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_NAMES = {"real": 0, "fake": 1, "ai": 1}
SPLIT_NAMES = {"train", "test", "val", "valid", "validation"}


@dataclass
class ImageRecord:
    path: str
    relative_path: str
    split: str
    label: int
    label_name: str
    width: int
    height: int
    mode: str
    image_format: str
    file_size: int
    file_sha256: str
    pixel_sha256: str
    dhash: str
    exact_duplicate_group: str = ""
    include: bool = True
    exclusion_reason: str = ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(image: Image.Image) -> str:
    """Hash decoded pixels so identical images in different formats match."""
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Return a small difference hash used only to produce review candidates."""
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


def _normalise_split(value: str) -> str:
    value = value.lower()
    return "val" if value in {"val", "valid", "validation"} else value


def _infer_metadata(path: Path, root: Path) -> tuple[str, int, str]:
    parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
    splits = [_normalise_split(part) for part in parts if part in SPLIT_NAMES]
    labels = [(LABEL_NAMES[part], "ai" if LABEL_NAMES[part] == 1 else "real") for part in parts if part in LABEL_NAMES]

    if len(set(splits)) != 1 or len(set(labels)) != 1:
        raise ValueError("path must contain exactly one split and one REAL/FAKE label directory")
    label, label_name = labels[0]
    return splits[0], label, label_name


def _read_record(path: Path, root: Path) -> ImageRecord:
    split, label, label_name = _infer_metadata(path, root)
    file_hash = _sha256_file(path)

    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as source:
            image_format = source.format or "unknown"
            mode = source.mode
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            rgb.load()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"unreadable image: {exc}") from exc

    if rgb.width < 1 or rgb.height < 1:
        raise ValueError("image has an invalid size")

    return ImageRecord(
        path=str(path.resolve()),
        relative_path=path.relative_to(root).as_posix(),
        split=split,
        label=label,
        label_name=label_name,
        width=rgb.width,
        height=rgb.height,
        mode=mode,
        image_format=image_format,
        file_size=path.stat().st_size,
        file_sha256=file_hash,
        pixel_sha256=_pixel_sha256(rgb),
        dhash=_dhash(rgb),
    )


def _apply_exact_duplicate_policy(records: list[ImageRecord]) -> dict[str, int]:
    """Exclude leakage while preferring the official test representative.

    Cross-label duplicate groups are entirely excluded for label review. For a
    same-label group spanning splits, one test image is retained and all other
    copies are excluded. Otherwise, the first path is retained.
    """
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.pixel_sha256].append(record)

    stats = Counter()
    duplicate_index = 0
    for members in groups.values():
        if len(members) == 1:
            continue

        duplicate_index += 1
        group_id = f"exact_{duplicate_index:06d}"
        for member in members:
            member.exact_duplicate_group = group_id

        labels = {member.label for member in members}
        splits = {member.split for member in members}
        stats["exact_duplicate_groups"] += 1
        stats["exact_duplicate_images"] += len(members)

        if len(labels) > 1:
            stats["cross_label_groups"] += 1
            for member in members:
                member.include = False
                member.exclusion_reason = "exact_duplicate_label_conflict"
            continue

        if len(splits) > 1:
            stats["cross_split_groups"] += 1

        ordered = sorted(members, key=lambda item: (item.split != "test", item.relative_path))
        for duplicate in ordered[1:]:
            duplicate.include = False
            duplicate.exclusion_reason = "exact_duplicate"

    return dict(stats)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_cifake(
    input_dir: str | Path,
    output_dir: str | Path,
    progress_every: int = 0,
) -> dict:
    root = Path(input_dir).resolve()
    output = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CIFAKE input directory does not exist: {root}")

    candidates = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        raise ValueError(f"No supported image files found under {root}")

    if progress_every > 0:
        print(f"Found {len(candidates):,} image files. Validating...", file=sys.stderr, flush=True)

    records: list[ImageRecord] = []
    rejected: list[dict[str, str]] = []
    for index, path in enumerate(candidates, start=1):
        try:
            records.append(_read_record(path, root))
        except ValueError as exc:
            rejected.append({
                "path": str(path.resolve()),
                "relative_path": path.relative_to(root).as_posix(),
                "reason": str(exc),
            })
        if progress_every > 0 and (index % progress_every == 0 or index == len(candidates)):
            print(
                f"  scanned {index:,}/{len(candidates):,} "
                f"(valid {len(records):,}, rejected {len(rejected):,})",
                file=sys.stderr,
                flush=True,
            )

    duplicate_stats = _apply_exact_duplicate_policy(records)
    output.mkdir(parents=True, exist_ok=True)

    record_rows = [asdict(record) for record in records]
    fields = list(ImageRecord.__dataclass_fields__)
    _write_csv(output / "manifest.csv", record_rows, fields)
    _write_csv(
        output / "clean_manifest.csv",
        [row for row in record_rows if row["include"]],
        fields,
    )
    _write_csv(output / "rejected.csv", rejected, ["path", "relative_path", "reason"])

    # Same dHash is a review signal, not proof of duplication.
    dhash_groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        dhash_groups[record.dhash].append(record)
    review_rows = []
    review_group_count = 0
    for dhash, members in sorted(dhash_groups.items()):
        if len(members) < 2 or len({item.pixel_sha256 for item in members}) == 1:
            continue
        review_group_count += 1
        review_id = f"review_{review_group_count:06d}"
        for member in members:
            review_rows.append({
                "review_group": review_id,
                "dhash": dhash,
                "relative_path": member.relative_path,
                "split": member.split,
                "label": member.label,
            })
    _write_csv(
        output / "near_duplicate_review.csv",
        review_rows,
        ["review_group", "dhash", "relative_path", "split", "label"],
    )

    included = [record for record in records if record.include]
    report = {
        "input_dir": str(root),
        "output_dir": str(output),
        "files_discovered": len(candidates),
        "valid_images": len(records),
        "included_images": len(included),
        "rejected_images": len(rejected),
        "counts_by_split_and_label": dict(sorted(Counter(
            f"{record.split}/{record.label_name}" for record in included
        ).items())),
        "dimensions": dict(sorted(Counter(
            f"{record.width}x{record.height}" for record in records
        ).items())),
        "near_duplicate_review_groups": review_group_count,
        **duplicate_stats,
    }
    with (output / "cleaning_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if progress_every > 0:
        print(f"Done. Reports written to {output}", file=sys.stderr, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CIFAKE and build clean manifests.")
    parser.add_argument("--input", default="data/raw/cifake", help="Raw CIFAKE directory.")
    parser.add_argument("--output", default="data/processed/cifake", help="Manifest output directory.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N files (0 disables progress output).",
    )
    args = parser.parse_args()

    report = clean_cifake(args.input, args.output, progress_every=args.progress_every)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
