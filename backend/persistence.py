"""Persistence adapters for feed metadata."""

from __future__ import annotations

from threading import RLock
from typing import Any

from backend.config import supabase_enabled, supabase_settings


def _supabase_client():
    from supabase import Client, create_client

    url, key, _ = supabase_settings()
    if not url or not key:
        raise RuntimeError("Supabase requires SUPABASE_URL and SUPABASE_SECRET_KEY")
    client: Client = create_client(url, key)
    return client


def _row(post: dict[str, Any], *, is_seed: bool = False) -> dict[str, Any]:
    return {
        "image_id": post["image_id"],
        "account_id": post["account_id"],
        "handle": post["handle"],
        "thumbnail_url": post["thumbnail_url"],
        "ai_probability": post["ai_probability"],
        "confidence": post["confidence"],
        "robustness": post["robustness"],
        "similarity_cluster": post["similarity_cluster"],
        "repetition_score": post["repetition_score"],
        "diversity_label": post["diversity_label"],
        "created_at": post["created_at"],
        "analysis_mode": post.get("analysis_mode", "mock"),
        "source": post.get("source", "upload"),
        "image_sha256": post.get("image_sha256", ""),
        "width": post.get("width"),
        "height": post.get("height"),
        "image_format": post.get("image_format"),
        "image_storage_key": post.get("image_storage_key"),
        "thumbnail_storage_key": post.get("thumbnail_storage_key"),
        "is_seed": is_seed,
    }


class LocalRepository:
    mode = "local"

    def __init__(self) -> None:
        self.lock = RLock()
        self.posts: dict[str, dict[str, Any]] = {}
        self.feed_ids: list[str] = []
        self.next_cluster_id = 1

    def initialize(self, seed_posts: list[dict[str, Any]]) -> None:
        self.reset(seed_posts)

    def reset(self, seed_posts: list[dict[str, Any]]) -> None:
        with self.lock:
            self.posts = {post["image_id"]: post for post in seed_posts}
            self.feed_ids = [post["image_id"] for post in seed_posts]
            max_cluster = max((post["similarity_cluster"] for post in seed_posts), default=0)
            self.next_cluster_id = max_cluster + 1

    def all_posts(self) -> list[dict[str, Any]]:
        with self.lock:
            posts = [self.posts[image_id] for image_id in self.feed_ids if image_id in self.posts]
            return sorted(posts, key=lambda post: post.get("created_at", ""), reverse=True)

    def add_posts(self, posts: list[dict[str, Any]]) -> None:
        with self.lock:
            for post in posts:
                self.posts[post["image_id"]] = post
                self.feed_ids.insert(0, post["image_id"])

    def update_posts(self, posts: list[dict[str, Any]]) -> None:
        return None

    def next_cluster(self) -> int:
        with self.lock:
            cluster = self.next_cluster_id
            self.next_cluster_id += 1
            return cluster

    def find_by_hash(self, image_sha256: str) -> list[dict[str, Any]]:
        with self.lock:
            return [post for post in self.posts.values() if post.get("image_sha256") == image_sha256]


class SupabaseRepository:
    mode = "supabase"

    def __init__(self) -> None:
        self.client = _supabase_client()
        self.lock = RLock()

    @staticmethod
    def _post(data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)
        data.pop("is_seed", None)
        return data

    def initialize(self, seed_posts: list[dict[str, Any]]) -> None:
        if seed_posts:
            self.client.table("posts").upsert(
                [_row(post, is_seed=True) for post in seed_posts],
                on_conflict="image_id",
            ).execute()

    def reset(self, seed_posts: list[dict[str, Any]]) -> None:
        self.client.table("posts").delete().eq("is_seed", False).execute()
        if seed_posts:
            self.client.table("posts").upsert(
                [_row(post, is_seed=True) for post in seed_posts],
                on_conflict="image_id",
            ).execute()

    def all_posts(self) -> list[dict[str, Any]]:
        response = self.client.table("posts").select("*").order("created_at", desc=True).execute()
        return [self._post(item) for item in (response.data or [])]

    def add_posts(self, posts: list[dict[str, Any]]) -> None:
        if posts:
            self.client.table("posts").insert([_row(post) for post in posts]).execute()

    def update_posts(self, posts: list[dict[str, Any]]) -> None:
        for post in posts:
            self.client.table("posts").update({
                "repetition_score": post["repetition_score"],
                "diversity_label": post["diversity_label"],
            }).eq("image_id", post["image_id"]).execute()

    def next_cluster(self) -> int:
        response = self.client.table("posts").select("similarity_cluster").order(
            "similarity_cluster", desc=True
        ).limit(1).execute()
        highest = (response.data or [{}])[0].get("similarity_cluster", 0)
        return int(highest or 0) + 1

    def find_by_hash(self, image_sha256: str) -> list[dict[str, Any]]:
        response = self.client.table("posts").select("*").eq("image_sha256", image_sha256).execute()
        return [self._post(item) for item in (response.data or [])]


REPOSITORY: LocalRepository | SupabaseRepository = (
    SupabaseRepository() if supabase_enabled() else LocalRepository()
)
