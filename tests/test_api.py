import pytest
from fastapi.testclient import TestClient
from nebula.api.server import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_api_tokenize(client):
    res = client.post("/api/tokenize", json={"text": "Test tokenize"})
    assert res.status_code == 200
    data = res.json()
    assert data["total_tokens"] > 0


def test_api_inference(client):
    res = client.post("/api/inference", json={"prompt": "Nebula", "max_new_tokens": 2})
    assert res.status_code == 200
    data = res.json()
    assert "generated_text" in data
    assert len(data["frames"]) > 0
