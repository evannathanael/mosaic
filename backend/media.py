"""Image storage adapters for local development and Supabase Storage."""

from __future__ import annotations

from pathlib import Path

from backend.config import supabase_enabled, supabase_settings
from backend.persistence import _supabase_client


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "backend" / "uploads"


class LocalMediaStore:
    mode = "local"

    def save(self, image_id: str, suffix: str, mime_type: str, original: bytes, thumbnail: bytes) -> dict[str, str]:
        originals_dir = UPLOAD_DIR / "originals"
        thumbnails_dir = UPLOAD_DIR / "thumbnails"
        originals_dir.mkdir(parents=True, exist_ok=True)
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        original_name = f"originals/{image_id}{suffix}"
        thumb_name = f"thumbnails/{image_id}.jpg"
        (UPLOAD_DIR / original_name).write_bytes(original)
        (UPLOAD_DIR / thumb_name).write_bytes(thumbnail)
        return {
            "thumbnail_url": f"/static/uploads/{thumb_name}",
            "image_storage_key": original_name,
            "thumbnail_storage_key": thumb_name,
        }


class SupabaseMediaStore:
    mode = "supabase"

    def __init__(self) -> None:
        self.client = _supabase_client()
        _, _, self.bucket = supabase_settings()

    def save(self, image_id: str, suffix: str, mime_type: str, original: bytes, thumbnail: bytes) -> dict[str, str]:
        original_key = f"originals/{image_id}{suffix}"
        thumb_key = f"thumbnails/{image_id}.jpg"
        bucket = self.client.storage.from_(self.bucket)
        bucket.upload(original_key, original, file_options={"content-type": mime_type, "upsert": "false"})
        bucket.upload(thumb_key, thumbnail, file_options={"content-type": "image/jpeg", "upsert": "false"})
        return {
            "thumbnail_url": bucket.get_public_url(thumb_key),
            "image_storage_key": original_key,
            "thumbnail_storage_key": thumb_key,
        }


MEDIA_STORE: LocalMediaStore | SupabaseMediaStore = (
    SupabaseMediaStore() if supabase_enabled() else LocalMediaStore()
)
