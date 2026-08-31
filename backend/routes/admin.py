"""Administrative demo reset endpoint."""

from fastapi import APIRouter

from backend.demo import load_seed_posts
from backend.routes.upload import apply_seed_scores_and_labels
from backend.similarity_index import attach_seed_embeddings
from backend.state import all_posts, reset


router = APIRouter()


@router.get("/api/debug/embeddings")
def debug_embeddings() -> dict:
    """Inspect which posts have a CLIP embedding and how they were clustered."""
    posts = all_posts()
    rows = [
        {
            "image_id": p["image_id"],
            "handle": p["handle"],
            "source": p.get("source"),
            "similarity_cluster": p["similarity_cluster"],
            "has_embedding": bool(p.get("embedding")),
            "embedding_dim": len(p["embedding"]) if p.get("embedding") else 0,
            "sha256": (p.get("image_sha256") or "")[:12],
        }
        for p in posts
    ]
    with_emb = sum(r["has_embedding"] for r in rows)
    return {
        "total": len(rows),
        "with_embedding": with_emb,
        "without_embedding": len(rows) - with_emb,
        "posts": rows,
    }


@router.post("/api/reset")
@router.post("/reset")
def reset_demo() -> dict[str, str]:
    seed = load_seed_posts()
    reset(seed)
    attach_seed_embeddings(all_posts())
    apply_seed_scores_and_labels()
    return {"status": "reset"}
