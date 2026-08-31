"""Precompute the trained detector's P(AI) for every image served by the demo backend.

Writes ``data/dataset/ai_scores.npz`` (paths relative to ``data/dataset/`` +
matching float32 P(AI) scores). The backend loads this at startup
(backend/detector.py:attach_seed_scores) so seed posts get a real model score
without paying the CLIP cost at request time.

Run once after changing the committed demo images or the detector checkpoint
(needs torch + open_clip; downloads the CLIP ViT-L/14 weights on first run):

    python scripts/build_demo_ai_scores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from backend.detector import predict_pil, _load_detector  # noqa: E402

DATASET_DIR = ROOT / "data" / "dataset"
OUT = DATASET_DIR / "ai_scores.npz"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main() -> None:
    if not DATASET_DIR.exists():
        raise SystemExit(f"{DATASET_DIR} not found")
    if _load_detector() is None:
        raise SystemExit("Detector checkpoint not loadable — see configs/config.yaml detector.checkpoint")

    paths, scores = [], []
    for path in sorted(DATASET_DIR.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with Image.open(path) as img:
            prob = predict_pil(img)
        if prob is None:
            print(f"  skip (scoring failed): {path}")
            continue
        rel = path.relative_to(DATASET_DIR).as_posix()
        paths.append(rel)
        scores.append(np.float32(prob))

    if not paths:
        raise SystemExit(f"No images ({', '.join(IMAGE_EXTENSIONS)}) under {DATASET_DIR}")

    np.savez_compressed(OUT, paths=np.array(paths), scores=np.array(scores, dtype="float32"))

    arr = np.array(scores)
    real = [s for p, s in zip(paths, scores) if p.startswith("REAL/")]
    fake = [s for p, s in zip(paths, scores) if p.startswith("FAKE/")]
    print(f"Wrote {len(paths)} scores -> {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  overall mean P(AI): {arr.mean():.3f}")
    if real:
        print(f"  REAL/ mean P(AI): {np.mean(real):.3f}  (want low)")
    if fake:
        print(f"  FAKE/ mean P(AI): {np.mean(fake):.3f}  (want high)")


if __name__ == "__main__":
    main()
