"""Load the small, committed demo image set into backend seed posts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "dataset"
SEED_FILE = ROOT / "data" / "demo_seed.json"


def _base_post(image_id: str, account_id: str, handle: str, url: str, probability: float,
               cluster: int, repetition: float, label: str, source: str,
               image_sha256: str = "", created_at: str | None = None) -> dict:
    return {
        "image_id": image_id,
        "account_id": account_id,
        "handle": handle,
        "thumbnail_url": url,
        "ai_probability": probability,
        "confidence": min(1.0, abs(probability - 0.5) * 2),
        "robustness": {
            "compressed": max(0.0, probability - 0.02),
            "cropped": max(0.0, probability - 0.05),
            "blurred": max(0.0, probability - 0.08),
            "recolored": max(0.0, probability - 0.03),
        },
        "similarity_cluster": cluster,
        "repetition_score": repetition,
        "diversity_label": label,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "analysis_mode": "demo_fixture",
        "source": source,
        "image_sha256": image_sha256,
    }


def _load_json_seed() -> list[dict] | None:
    if not SEED_FILE.exists():
        return None
    try:
        payload = json.loads(SEED_FILE.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload:
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("posts"), list):
            return payload["posts"]
    except (OSError, json.JSONDecodeError):
        return None
    return None


def load_seed_posts() -> list[dict]:
    supplied = _load_json_seed()
    if supplied:
        return supplied
    if not DATASET_DIR.exists():
        return []

    real = sorted((DATASET_DIR / "REAL").glob("*")) if (DATASET_DIR / "REAL").exists() else []
    fake = sorted((DATASET_DIR / "FAKE").glob("*")) if (DATASET_DIR / "FAKE").exists() else []
    fake_images = [path for path in fake if path.is_file()]
    posts: list[dict] = []
    demo_paths = [(path, False) for path in real if path.is_file()]
    demo_paths.extend((path, True) for path in fake_images)
    start = datetime.now(timezone.utc) - timedelta(minutes=len(demo_paths))
    flood_count = max(1, min(len(fake_images), round(len(fake_images) * 0.7)))
    flood_index = 0
    fake_index = 0
    # The fixture deliberately contains a flooding AI account and a distinct
    # AI creator account so the UI can demonstrate both product outcomes.
    for index, (path, is_fake) in enumerate(demo_paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(DATASET_DIR).as_posix()
        created_at = (start + timedelta(minutes=index)).isoformat()
        if not is_fake:
            posts.append(_base_post(
                f"demo_creator_real_{index + 1}", "demo-creator", "@oneoff.studio",
                f"/static/demo/{quote(rel, safe='/')}", 0.08, 300 + index, 0.05,
                "original", "demo_dataset", digest, created_at,
            ))
            continue
        is_flood = flood_index < flood_count
        if is_flood:
            repetition = min(0.94, 0.35 + flood_index * 0.12)
            label = "repeated_synthetic" if repetition >= 0.75 else "unique_ai"
            account_id, handle, cluster = "demo-flood", "@synthetic.flood", 200
            flood_index += 1
        else:
            repetition, label = 0.08, "unique_ai"
            account_id, handle, cluster = "demo-creator", "@oneoff.studio", 400 + fake_index
        posts.append(_base_post(
            f"demo_ai_{fake_index + 1}", account_id, handle,
            f"/static/demo/{quote(rel, safe='/')}", 0.94, cluster, repetition, label,
            "demo_dataset", digest, created_at,
        ))
        fake_index += 1
    return posts
