"""Feed, cluster, account-summary, and simulation endpoints."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.schemas import (
    AccountSummary,
    Cluster,
    ClusterDetail,
    FeedPost,
    FeedSimulationRequest,
    FeedSimulationResponse,
    Post,
)
from backend.state import all_posts


router = APIRouter()


def _uploaded_at(post: dict) -> int:
    value = post.get("uploaded_at")
    if value is not None:
        return int(value)
    try:
        return int(datetime.fromisoformat(post["created_at"].replace("Z", "+00:00")).timestamp())
    except (KeyError, ValueError, TypeError):
        return int(datetime.now(timezone.utc).timestamp())


def _feed_post(post: dict) -> dict:
    return {
        "image_id": post["image_id"],
        "thumbnail_url": post["thumbnail_url"],
        "handle": post["handle"],
        "ai_probability": post["ai_probability"],
        "repetition_score": post["repetition_score"],
        "diversity_label": post["diversity_label"],
        "uploaded_at": _uploaded_at(post),
    }


@router.get("/api/feed", response_model=list[FeedPost])
@router.get("/feed", response_model=list[FeedPost])
def feed() -> list[dict]:
    return [_feed_post(post) for post in all_posts()]


@router.get("/api/clusters", response_model=list[Cluster])
def clusters() -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for post in all_posts():
        grouped[post["similarity_cluster"]].append(post)
    return [
        {
            "similarity_cluster": cluster_id,
            "representative_image_id": members[0]["image_id"],
            "count": len(members),
            "image_ids": [member["image_id"] for member in members],
            "average_similarity": 1.0,
        }
        for cluster_id, members in sorted(grouped.items())
    ]


@router.get("/api/cluster/{cluster_id}", response_model=ClusterDetail)
@router.get("/cluster/{cluster_id}", response_model=ClusterDetail)
def cluster_detail(cluster_id: int) -> dict:
    members = [post for post in all_posts() if post["similarity_cluster"] == cluster_id]
    if not members:
        raise HTTPException(status_code=404, detail="Cluster not found")
    public_members = [_feed_post(post) for post in members]
    kept = public_members[0]["image_id"]
    return {
        "cluster_id": cluster_id,
        "kept_image_id": kept,
        "members": public_members,
        "suppressed_image_ids": [member["image_id"] for member in public_members[1:]],
    }


@router.get("/api/account/{account_id}/summary", response_model=AccountSummary)
def account_summary(account_id: str) -> dict:
    posts = [post for post in all_posts() if post["account_id"] == account_id]
    flagged = [post for post in posts if post["diversity_label"] == "repeated_synthetic"]
    trend = [
        {"date": post["created_at"][:10], "repetition_score": post["repetition_score"]}
        for post in sorted(posts, key=lambda item: item["created_at"])
    ]
    return {
        "account_id": account_id,
        "total_uploads": len(posts),
        "flagged_percentage": round((len(flagged) / len(posts)) * 100, 2) if posts else 0.0,
        "repeated_synthetic_count": len(flagged),
        "repetition_trend": trend,
    }


@router.post("/api/simulate-feed", response_model=FeedSimulationResponse)
def simulate_feed(request: FeedSimulationRequest) -> dict:
    before = request.images
    seen_groups: set[tuple[str, str]] = set()
    after: list[FeedPost] = []
    suppressed: list[str] = []
    for image in before:
        repeated = image.diversity_label == "repeated_synthetic"
        group = (image.handle, image.diversity_label)
        if repeated and group in seen_groups:
            suppressed.append(image.image_id)
            continue
        after.append(image)
        if repeated:
            seen_groups.add(group)
    return {"before": before, "after": after, "suppressed_image_ids": suppressed}
