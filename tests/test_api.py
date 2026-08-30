import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.server import app


def _jpeg(color=(20, 40, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_and_seed_feed():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is False
        feed = client.get("/api/feed")
        assert feed.status_code == 200
        assert feed.json()
        assert client.get(feed.json()[0]["thumbnail_url"]).status_code == 200


def test_upload_and_duplicate_cluster():
    payload = _jpeg()
    with TestClient(app) as client:
        response = client.post("/api/analyze", files={"files": ("photo.jpg", payload, "image/jpeg")})
        assert response.status_code == 200
        first = response.json()["items"][0]
        assert first["analysis_mode"] == "mock"
        assert first["ai_probability"] == 0.5

        second = client.post("/api/upload", files={"file": ("photo.jpg", payload, "image/jpeg")})
        assert second.status_code == 200
        assert second.json()["similarity_cluster"] == first["similarity_cluster"]
        assert client.get("/api/clusters").status_code == 200


def test_upload_rejects_invalid_file():
    with TestClient(app) as client:
        response = client.post("/api/upload", files={"file": ("bad.jpg", b"not image", "image/jpeg")})
        assert response.status_code == 400


def test_simulate_feed_suppresses_repeated_cluster():
    with TestClient(app) as client:
        feed = client.get("/api/feed").json()
        response = client.post("/api/simulate-feed", json={"images": feed})
        assert response.status_code == 200
        body = response.json()
        assert len(body["after"]) <= len(body["before"])


def test_frontend_contract_and_reset():
    with TestClient(app) as client:
        feed = client.get("/feed")
        assert feed.status_code == 200
        posts = feed.json()
        assert posts
        assert set(posts[0]) == {
            "image_id", "thumbnail_url", "handle", "ai_probability",
            "repetition_score", "diversity_label", "uploaded_at",
        }
        assert all(posts[i]["uploaded_at"] >= posts[i + 1]["uploaded_at"] for i in range(len(posts) - 1))

        response = client.post("/upload", files={"file": ("ai.png", _jpeg(), "image/jpeg")})
        assert response.status_code == 200
        assert set(response.json()) == set(posts[0])
        assert isinstance(response.json()["uploaded_at"], int)

        assert client.post("/reset").json() == {"status": "reset"}


def test_cluster_detail_contract():
    with TestClient(app) as client:
        clusters = client.get("/api/clusters").json()
        response = client.get(f"/cluster/{clusters[0]['similarity_cluster']}")
        assert response.status_code == 200
        body = response.json()
        assert body["kept_image_id"] == body["members"][0]["image_id"]
        assert body["suppressed_image_ids"] == [member["image_id"] for member in body["members"][1:]]
