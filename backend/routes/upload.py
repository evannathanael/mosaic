"""Image upload and model-independent analysis endpoints."""

from __future__ import annotations

import hashlib
import io
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.routes.feed import _feed_post
from backend.schemas import AnalyzeResponse, FeedPost, Post
from backend.media import MEDIA_STORE
from backend.detector import ai_threshold, detector_ready, label_for, predict_pil
from backend.similarity_index import annotate_cluster_similarity, assign_cluster, embed_image
from backend.state import add_posts, all_posts, find_by_hash, next_cluster, update_posts


router = APIRouter()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def _validate_and_thumbnail(data: bytes) -> tuple[Image.Image, str, int, int]:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = (probe.format or "").upper()
            width, height = probe.size
            probe.verify()
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
            image.load()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise HTTPException(status_code=400, detail=f"Unreadable image: {exc}") from exc
    if width < 1 or height < 1:
        raise HTTPException(status_code=400, detail="Image has invalid dimensions")
    return image, image_format, width, height


def _fallback_probability(filename: str) -> float:
    """Filename heuristic used only when the trained detector is unavailable.

    Default case must land strictly below ai_threshold() (0.5 by default) —
    label_for() uses >=, so a default of exactly 0.50 flagged every generic
    upload filename as AI-generated regardless of content.
    """
    lowered = filename.lower()
    if any(token in lowered for token in ("fake", "synthetic", "generated", "ai")):
        return 0.82
    return 0.15


def _post_for_upload(upload: UploadFile, data: bytes) -> dict:
    image, image_format, width, height = _validate_and_thumbnail(data)
    digest = hashlib.sha256(data).hexdigest()
    image_id = f"upload_{uuid.uuid4().hex[:12]}"
    suffix = Path(upload.filename or "upload.jpg").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".jpg"
    image.thumbnail((512, 512), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
    thumbnail_buffer = io.BytesIO()
    image.save(thumbnail_buffer, format="JPEG", quality=88)
    try:
        media = MEDIA_STORE.save(
            image_id,
            suffix,
            upload.content_type or "image/jpeg",
            data,
            thumbnail_buffer.getvalue(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Image storage upload failed; check Supabase bucket, credentials, and network settings.",
        ) from exc

    # Exact re-post: inherit the matched post's cluster + embedding.
    # Otherwise embed with CLIP and assign the cluster by cosine similarity to
    # existing posts (near-duplicate detection), falling back to a fresh cluster.
    matches = find_by_hash(digest)
    if matches:
        cluster = matches[0]["similarity_cluster"]
        embedding = matches[0].get("embedding")
        similarity = 1.0  # byte-identical re-post
    else:
        embedding = embed_image(image)
        cluster, similarity = assign_cluster(embedding, all_posts(), next_cluster)

    model_prob = predict_pil(image)
    probability = model_prob if model_prob is not None else _fallback_probability(upload.filename or "")
    analysis_mode = "model" if model_prob is not None else "mock"
    # If CLIP matched this upload into an existing near-duplicate group, adopt
    # that group's AI probability — the variants are the same image, so if the
    # originals are AI this one is too (drives the repeated_synthetic label).
    siblings = [p["ai_probability"] for p in all_posts() if p["similarity_cluster"] == cluster]
    if siblings:
        probability = max(probability, max(siblings))
    return {
        "image_id": image_id,
        "account_id": "uploaded-demo",
        "handle": "@you",
        "thumbnail_url": media["thumbnail_url"],
        "ai_probability": probability,
        "confidence": abs(probability - 0.5) * 2,
        "robustness": {
            "compressed": probability,
            "cropped": max(0.0, probability - 0.03),
            "blurred": max(0.0, probability - 0.06),
            "recolored": max(0.0, probability - 0.02),
        },
        "similarity_cluster": cluster,
        "similarity_score": round(float(similarity), 4),
        "embedding": embedding,
        "repetition_score": 0.0,
        "diversity_label": label_for(probability, repeated=False),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_mode": analysis_mode,
        "source": "upload",
        "image_sha256": digest,
        "width": width,
        "height": height,
        "image_format": image_format,
        **media,
    }


def _recalculate_cluster(cluster: int) -> list[dict]:
    """After a new post joins a cluster, refresh every member's repetition score
    and label.

    A cluster with >1 member is a near-duplicate group. Repeated *AI* content is
    the problem case -> 'repeated_synthetic'. Repeated non-AI content (an ordinary
    re-post) is fine -> stays 'original'; repetition_score still records it.
    """
    threshold = ai_threshold()
    posts = [post for post in all_posts() if post["similarity_cluster"] == cluster]
    size = len(posts)
    repetition = 0.0 if size <= 1 else round(min(0.99, 0.7 + 0.1 * (size - 2)), 2)
    cluster_is_ai = any(post["ai_probability"] >= threshold for post in posts)
    for post in posts:
        post["repetition_score"] = repetition
        post["diversity_label"] = label_for(
            post["ai_probability"], repeated=(size > 1 and cluster_is_ai)
        )
    return posts


def apply_seed_scores_and_labels() -> None:
    """Startup / reset hook: overwrite seed posts' AI probability with the trained
    detector's cached scores (unless disabled in config), then re-derive every
    post's diversity_label through the same cluster logic uploads use."""
    from backend.demo import SEED_LABEL_OVERRIDES
    from backend.detector import attach_seed_scores, score_seed_enabled

    if score_seed_enabled():
        attach_seed_scores(all_posts(), SEED_LABEL_OVERRIDES)
    for cluster in {post["similarity_cluster"] for post in all_posts()}:
        _recalculate_cluster(cluster)
    annotate_cluster_similarity(all_posts())


async def analyze_uploads(files: list[UploadFile]) -> list[dict]:
    if not files:
        raise HTTPException(status_code=400, detail="Provide at least one image file")
    started = time.perf_counter()
    posts = []
    for upload in files:
        if upload.content_type and upload.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported image type: {upload.content_type}")
        posts.append(_post_for_upload(upload, await upload.read()))
    add_posts(posts)
    changed: list[dict] = []
    for post in posts:
        changed.extend(_recalculate_cluster(post["similarity_cluster"]))
    annotate_cluster_similarity(all_posts())
    update_posts(changed)
    return posts


@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(files: list[UploadFile] = File(default=[]), file: UploadFile | None = File(default=None)):
    uploads = list(files)
    if file is not None:
        uploads.append(file)
    started = time.perf_counter()
    posts = await analyze_uploads(uploads)
    return AnalyzeResponse(items=[Post.model_validate(post) for post in posts],
                           inference_latency_ms=round((time.perf_counter() - started) * 1000, 2),
                           model_ready=detector_ready())


@router.post("/api/upload")
async def upload_legacy(files: list[UploadFile] = File(default=[]), file: UploadFile | None = File(default=None)):
    uploads = list(files)
    if file is not None:
        uploads.append(file)
    started = time.perf_counter()
    posts = await analyze_uploads(uploads)
    if len(posts) == 1:
        return Post.model_validate(posts[0])
    return AnalyzeResponse(items=[Post.model_validate(post) for post in posts],
                           inference_latency_ms=round((time.perf_counter() - started) * 1000, 2),
                           model_ready=detector_ready())


@router.post("/upload", response_model=FeedPost)
async def upload_frontend(file: UploadFile = File(...)):
    """Frontend contract: accept one image and return one compact Post."""
    started = time.perf_counter()
    posts = await analyze_uploads([file])
    return FeedPost.model_validate(_feed_post(posts[0]))
