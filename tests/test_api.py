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
        body = health.json()
        assert isinstance(body["model_ready"], bool)
        assert body["analysis_mode"] == ("model" if body["model_ready"] else "mock")
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
        assert first["analysis_mode"] in {"model", "mock"}
        assert 0.0 <= first["ai_probability"] <= 1.0
        assert response.json()["model_ready"] is (first["analysis_mode"] == "model")

        second = client.post("/api/upload", files={"file": ("photo.jpg", payload, "image/jpeg")})
        assert second.status_code == 200
        assert second.json()["similarity_cluster"] == first["similarity_cluster"]
        assert client.get("/api/clusters").status_code == 200


def test_feed_exposes_similarity_cluster():
    with TestClient(app) as client:
        feed = client.get("/feed").json()
        assert all(isinstance(post["similarity_cluster"], int) for post in feed)
        assert all(isinstance(post["cluster_size"], int) and post["cluster_size"] >= 1 for post in feed)
        assert all(0.0 <= post["similarity_score"] <= 1.0 for post in feed)


def test_cluster_size_matches_membership():
    with TestClient(app) as client:
        feed = client.get("/feed").json()
        from collections import Counter

        actual = Counter(p["similarity_cluster"] for p in feed)
        for post in feed:
            assert post["cluster_size"] == actual[post["similarity_cluster"]]


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
            "repetition_score", "diversity_label", "similarity_cluster",
            "cluster_size", "similarity_score", "uploaded_at",
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
