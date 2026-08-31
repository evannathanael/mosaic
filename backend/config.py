"""Runtime configuration for the backend."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@lru_cache(maxsize=1)
def supabase_settings() -> tuple[str | None, str | None, str]:
    return (
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        os.getenv("SUPABASE_STORAGE_BUCKET", "mosaic-images"),
    )


def supabase_enabled() -> bool:
    url, key, _ = supabase_settings()
    return bool(url and key)
