"""Generate committed near-duplicate fixtures for the demo feed.

The demo images in ``data/dataset/`` are all visually distinct, so the feed
diversity ranker has nothing to space apart. This script takes the app's own
``data/dataset/FAKE/`` images and produces a handful of near-duplicate variants
per image (repost-style crop / recolor / recompress / resize edits).

Cluster membership is known by construction — each source image and its variants
share one integer ``similarity_cluster`` — so no CLIP / embedding run is needed.
Pure Pillow, so it runs in the same lightweight env as the backend.

Output (all committed, served by the existing ``/static/demo`` mount):
    data/dataset/near_dupes/cluster_<NNNN>/original.jpg
    data/dataset/near_dupes/cluster_<NNNN>/variant_<NN>.jpg
    data/dataset/near_dupes/manifest.csv   (image_path,similarity_cluster)

Usage:
    python scripts/build_demo_dupes.py [--n-variants 5] [--max-edge 640]
"""
from __future__ import annotations

import argparse
import csv
import io
import random
from pathlib import Path

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "dataset" / "FAKE"
OUT_DIR = ROOT / "data" / "dataset" / "near_dupes"
DATASET_DIR = ROOT / "data" / "dataset"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _fit(img: Image.Image, max_edge: int) -> Image.Image:
    longest = max(img.size)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)


def _recompress(img: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=random.choice([30, 45, 60]))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _center_crop(img: Image.Image) -> Image.Image:
    frac = random.uniform(0.75, 0.92)
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.LANCZOS)


def _resize_roundtrip(img: Image.Image) -> Image.Image:
    scale = random.choice([0.5, 0.35, 0.25])
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return small.resize((w, h), Image.Resampling.LANCZOS)


def _color_jitter(img: Image.Image) -> Image.Image:
    out = img
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        out = enhancer(out).enhance(random.uniform(0.82, 1.18))
    return out


VARIANT_OPS = (_recompress, _center_crop, _resize_roundtrip, _color_jitter)


def _make_variant(base: Image.Image) -> Image.Image:
    out = base
    for op in random.sample(VARIANT_OPS, k=random.choice([1, 2])):
        out = op(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-variants", type=int, default=5, help="Variants per source image.")
    parser.add_argument("--max-edge", type=int, default=512, help="Longest-side pixel cap.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    sources = sorted(p for p in SOURCE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not sources:
        raise SystemExit(f"No source images found in {SOURCE_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for cluster_id, src_path in enumerate(sources):
        cluster_dir = OUT_DIR / f"cluster_{cluster_id:04d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        base = _fit(Image.open(src_path).convert("RGB"), args.max_edge)

        members = [("original.jpg", base)]
        members += [(f"variant_{i:02d}.jpg", _make_variant(base)) for i in range(args.n_variants)]

        for name, img in members:
            out_path = cluster_dir / name
            img.convert("RGB").save(out_path, format="JPEG", quality=85)
            rows.append({
                "image_path": out_path.relative_to(DATASET_DIR).as_posix(),
                "similarity_cluster": cluster_id,
            })

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "similarity_cluster"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(sources)} clusters / {len(rows)} images -> {OUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
