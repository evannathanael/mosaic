"""Administrative demo reset endpoint."""

from fastapi import APIRouter

from backend.demo import load_seed_posts
from backend.state import reset


router = APIRouter()


@router.post("/api/reset")
@router.post("/reset")
def reset_demo() -> dict[str, str]:
    seed = load_seed_posts()
    reset(seed)
    return {"status": "reset"}
