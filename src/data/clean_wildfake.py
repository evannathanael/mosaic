"""Clean a focused WildFake archive subset without extracting the full dataset.

The cleaner is intentionally archive-aware: it validates image members in ZIP
files and writes ``archive_path`` + ``member_path`` to the manifest.  This keeps
the raw archives immutable and avoids creating another multi-terabyte copy.

The default subset is:

* real: AFHQ, CelebA-HQ, Church
* synthetic: DDIM, DDPM

COCO val2017 and DALL-E Advanced are holdout data and are never accepted as
training archives.  Labels are binary and are derived only from the archive
source (0 = real, 1 = AI); no relabeling or model-based filtering is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
HOLDOUT_NAMES = {"coco", "dalle", "dall-e", "dalle3"}
DEFAULT_ARCHIVES = (
    "Images/Real/afhq.zip",
    "Images/Real/celebahq.zip",
    "Images/Real/church.zip",
    "Images/Diffusion_based/DDIM.zip",
    "Images/Diffusion_based/DDPM.zip",
)


@dataclass
class ImageRecord:
    archive_path: str
    member_path: str
    source: str
    generator: str
    split: str
    source_label: int
    is_advanced: int
    width: int
    height: int
    image_format: str
    image_mode: str
    image_sha256: str
    pixel_sha256: str = ""
    dhash: str = ""
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
    pixels = list(getattr(gray, "get_flattened_data", gray.getdata)())
    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            bits = (bits << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return f"{bits:0{hash_size * hash_size // 4}x}"


def _decode(raw: bytes, visual_hashes: bool = False):
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            image_format = probe.format or "unknown"
            image_mode = probe.mode
            width, height = probe.size
            probe.verify()
        image = None
        if visual_hashes:
            with Image.open(io.BytesIO(raw)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"unreadable image: {exc}") from exc
    if width < 1 or height < 1:
        raise ValueError("image has an invalid size")
    return width, height, image_format, image_mode, image


def _source_info(archive: Path) -> tuple[str, str, int, int]:
    """Return source name, generator, binary label, and advanced flag."""
    relative = archive.as_posix().lower()
    stem = archive.stem.lower()
    if stem in HOLDOUT_NAMES or "dalle" in relative or "coco" in relative:
        raise ValueError("holdout archive (COCO val2017/DALL-E Advanced) is excluded")
    if "/real/" in f"/{relative}/":
        return f"wildfake_real_{stem}", stem, 0, 0
    if "/diffusion_based/" in f"/{relative}/":
        return f"wildfake_{stem}", stem, 1, 0
    if "/gan_based" in f"/{relative}/" or "/other_based" in f"/{relative}/":
        return f"wildfake_{stem}", stem, 1, 0
    raise ValueError(f"cannot infer label from archive path: {archive}")


def _split(member_path: str) -> str:
    parts = {part.lower() for part in Path(member_path).parts}
    if "validation" in parts or "val" in parts:
        return "validation"
    if "test" in parts:
        return "test"
    if "train" in parts:
        return "train"
    # Some WildFake sources have no split in the archive. A stable path hash
    # gives a reproducible 80/10/10 split without random leakage.
    value = int(hashlib.sha256(member_path.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if value < 8 else "validation" if value == 8 else "test"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _apply_exact_duplicate_policy(records: list[ImageRecord]) -> dict[str, int]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.image_sha256].append(record)
    stats = Counter()
    group_number = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        group_number += 1
        group_id = f"exact_{group_number:06d}"
        for member in members:
            member.exact_duplicate_group = group_id
        stats["exact_duplicate_groups"] += 1
        stats["exact_duplicate_images"] += len(members)
        labels = {member.source_label for member in members}
        if len(labels) > 1:
            stats["cross_label_groups"] += 1
            for member in members:
                member.include = False
                member.exclusion_reason = "exact_duplicate_label_conflict"
            continue
        splits = {member.split for member in members}
        if len(splits) > 1:
            stats["cross_split_groups"] += 1
        # Keep validation/test copies when an exact duplicate crosses splits;
        # otherwise keep the first deterministic archive/member ordering.
        ordered = sorted(
            members,
            key=lambda item: (
                item.split not in {"validation", "test"},
                item.archive_path,
                item.member_path,
            ),
        )
        for duplicate in ordered[1:]:
            duplicate.include = False
            duplicate.exclusion_reason = "exact_duplicate"
    return dict(stats)


def clean_archives(
    input_dir: Path,
    output_dir: Path,
    archive_names: list[str],
    max_per_source: int | None = None,
    visual_hashes: bool = False,
    progress_every: int = 5000,
) -> dict:
    input_dir = input_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[ImageRecord] = []
    rejected: list[dict[str, str]] = []
    rejected_archives: list[dict[str, str]] = []
    report = Counter()
    report["archives_requested"] = len(archive_names)

    for archive_name in archive_names:
        archive = (input_dir / archive_name).resolve()
        try:
            source, generator, source_label, is_advanced = _source_info(archive.relative_to(input_dir))
        except (ValueError, RuntimeError) as exc:
            rejected_archives.append({"archive_path": archive_name, "reason": str(exc)})
            continue
        if not archive.exists():
            rejected_archives.append({"archive_path": archive_name, "reason": "archive_not_found"})
            continue
        if archive.name.lower().endswith(".incomplete"):
            rejected_archives.append({"archive_path": archive_name, "reason": "incomplete_archive"})
            continue
        try:
            with zipfile.ZipFile(archive) as zf:
                members = [
                    info for info in zf.infolist()
                    if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
                ]
                members.sort(key=lambda info: info.filename)
                if max_per_source and len(members) > max_per_source:
                    stride = len(members) / max_per_source
                    members = [members[min(int(i * stride), len(members) - 1)] for i in range(max_per_source)]
                report["image_members_discovered"] += len(members)
                for info in members:
                    member_path = info.filename.replace("\\", "/")
                    try:
                        raw = zf.read(info)
                        width, height, image_format, image_mode, image = _decode(raw, visual_hashes)
                        records.append(ImageRecord(
                            archive_path=archive.relative_to(input_dir).as_posix(),
                            member_path=member_path,
                            source=source,
                            generator=generator,
                            split=_split(member_path),
                            source_label=source_label,
                            is_advanced=is_advanced,
                            width=width,
                            height=height,
                            image_format=image_format,
                            image_mode=image_mode,
                            image_sha256=hashlib.sha256(raw).hexdigest(),
                            pixel_sha256=_pixel_sha256(image) if visual_hashes else "",
                            dhash=_dhash(image) if visual_hashes else "",
                        ))
                    except (OSError, ValueError, TypeError, zipfile.BadZipFile) as exc:
                        rejected.append({
                            "archive_path": archive.relative_to(input_dir).as_posix(),
                            "member_path": member_path,
                            "reason": str(exc),
                        })
                    report["rows_scanned"] += 1
                    if progress_every > 0 and report["rows_scanned"] % progress_every == 0:
                        print(
                            f"  scanned {report['rows_scanned']:,} images "
                            f"(valid {len(records):,}, rejected {len(rejected):,})",
                            file=sys.stderr,
                            flush=True,
                        )
        except (OSError, zipfile.BadZipFile) as exc:
            rejected_archives.append({"archive_path": archive_name, "reason": f"invalid_zip: {exc}"})

    report["valid_images"] = len(records)
    report.update(_apply_exact_duplicate_policy(records))
    included = [record for record in records if record.include]
    report["included_images"] = len(included)
    report["rejected_images"] = len(rejected)
    report["rejected_archives"] = len(rejected_archives)
    report["visual_hashes_enabled"] = visual_hashes
    report["max_per_source"] = max_per_source or 0
    report["counts_by_split_and_label"] = dict(Counter(
        f"{record.split}/label_{record.source_label}" for record in included
    ))
    report["counts_by_source"] = dict(Counter(record.source for record in included))

    fields = list(asdict(records[0]).keys()) if records else list(ImageRecord.__dataclass_fields__.keys())
    rows = [asdict(record) for record in records]
    clean_rows = [asdict(record) for record in included]
    _write_csv(output_dir / "manifest.csv", rows, fields)
    _write_csv(output_dir / "clean_manifest.csv", clean_rows, fields)
    _write_csv(output_dir / "rejected.csv", rejected, ["archive_path", "member_path", "reason"])
    _write_csv(output_dir / "rejected_archives.csv", rejected_archives, ["archive_path", "reason"])

    near_rows: list[dict[str, str]] = []
    if visual_hashes:
        by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
        for record in records:
            by_hash[record.dhash].append(record)
        for digest, members in by_hash.items():
            if digest and len(members) > 1:
                for member in members:
                    near_rows.append({
                        "dhash": digest,
                        "archive_path": member.archive_path,
                        "member_path": member.member_path,
                        "source_label": str(member.source_label),
                        "split": member.split,
                    })
    _write_csv(output_dir / "near_duplicate_review.csv", near_rows,
               ["dhash", "archive_path", "member_path", "source_label", "split"])
    with (output_dir / "cleaning_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"input_dir": str(input_dir), "output_dir": str(output_dir), **dict(report)}, handle, indent=2)
    return dict(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/wildfake"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/wildfake"))
    parser.add_argument("--archive", action="append", dest="archives", help="archive path relative to --input (repeatable)")
    parser.add_argument("--max-per-source", type=int, default=None)
    parser.add_argument("--visual-hashes", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()
    archives = args.archives or list(DEFAULT_ARCHIVES)
    report = clean_archives(args.input, args.output, archives, args.max_per_source, args.visual_hashes, args.progress_every)
    print(json.dumps(report, indent=2))
    if report.get("rejected_archives"):
        print("Warning: some requested archives were not processed; see rejected_archives.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
