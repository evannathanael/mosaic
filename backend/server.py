"""FastAPI server for the model-independent Mosaic demo."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.demo import DATASET_DIR, load_seed_posts
from backend.routes import admin, feed, upload
from backend.state import initialize, storage_mode


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize(load_seed_posts())
    yield


app = FastAPI(title="Mosaic API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(feed.router)
app.include_router(upload.router)
app.include_router(admin.router)
if DATASET_DIR.exists():
    app.mount("/static/demo", StaticFiles(directory=str(DATASET_DIR)), name="demo")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/api/health")
def health() -> dict[str, bool | str]:
    return {
        "status": "ok",
        "model_ready": False,
        "analysis_mode": "mock",
        "storage_mode": storage_mode(),
    }
