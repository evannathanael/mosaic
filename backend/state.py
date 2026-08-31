"""Repository-compatible state facade used by route modules."""

from __future__ import annotations

from typing import Any

from backend.persistence import REPOSITORY


def initialize(seed_posts: list[dict[str, Any]]) -> None:
    REPOSITORY.initialize(seed_posts)


def reset(seed_posts: list[dict[str, Any]]) -> None:
    REPOSITORY.reset(seed_posts)


def all_posts() -> list[dict[str, Any]]:
    return REPOSITORY.all_posts()


def add_posts(posts: list[dict[str, Any]]) -> None:
    REPOSITORY.add_posts(posts)


def update_posts(posts: list[dict[str, Any]]) -> None:
    REPOSITORY.update_posts(posts)


def next_cluster() -> int:
    return REPOSITORY.next_cluster()


def find_by_hash(image_sha256: str) -> list[dict[str, Any]]:
    return REPOSITORY.find_by_hash(image_sha256)


def storage_mode() -> str:
    return REPOSITORY.mode
