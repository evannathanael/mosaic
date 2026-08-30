"""API response/request models shared by Mosaic backend routes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RobustnessScores(BaseModel):
    compressed: float = 0.0
    cropped: float = 0.0
    blurred: float = 0.0
    recolored: float = 0.0


class Post(BaseModel):
    """Detailed internal/legacy response used by the analysis endpoints."""
    image_id: str
    account_id: str
    handle: str
    thumbnail_url: str
    ai_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    robustness: RobustnessScores
    similarity_cluster: int
    repetition_score: float = Field(ge=0.0, le=1.0)
    diversity_label: str
    created_at: str
    analysis_mode: str = "mock"
    source: str = "upload"


class FeedPost(BaseModel):
    """Exact post shape consumed by frontend/src/api.ts."""

    image_id: str
    thumbnail_url: str
    handle: str
    ai_probability: float = Field(ge=0.0, le=1.0)
    repetition_score: float = Field(ge=0.0, le=1.0)
    diversity_label: Literal["original", "unique_ai", "repeated_synthetic"]
    uploaded_at: int


class AnalyzeResponse(BaseModel):
    items: list[Post]
    inference_latency_ms: float
    model_ready: bool


class Cluster(BaseModel):
    similarity_cluster: int
    representative_image_id: str
    count: int
    image_ids: list[str]
    average_similarity: float = 1.0


class AccountSummary(BaseModel):
    account_id: str
    total_uploads: int
    flagged_percentage: float
    repeated_synthetic_count: int
    repetition_trend: list[dict[str, Any]]


class FeedSimulationRequest(BaseModel):
    images: list[FeedPost] = Field(default_factory=list)


class FeedSimulationResponse(BaseModel):
    before: list[FeedPost]
    after: list[FeedPost]
    suppressed_image_ids: list[str]


class ClusterDetail(BaseModel):
    cluster_id: int
    kept_image_id: str
    members: list[FeedPost]
    suppressed_image_ids: list[str]
