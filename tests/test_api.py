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


def test_api_backward(client):
    res = client.post("/api/backward", json={"prompt": "Nebula gradient", "target_token_id": 65})
    assert res.status_code == 200
    data = res.json()
    assert "loss" in data
    assert "layer_gradients" in data
    assert len(data["layer_gradients"]) > 0


def test_api_dla(client):
    res = client.post("/api/dla", json={"prompt": "Nebula circuit", "target_token_id": 65})
    assert res.status_code == 200
    data = res.json()
    assert "layer_attributions" in data
    assert "top_promoting_heads" in data


def test_api_steering(client):
    res_list = client.get("/api/steering/concepts")
    assert res_list.status_code == 200
    assert len(res_list.json()) > 0

    res_apply = client.post("/api/steering/apply", json={"prompt": "Test", "concept_name": "Reasoning", "intensity": 1.5})
    assert res_apply.status_code == 200
    data = res_apply.json()
    assert "steered_prediction" in data
    assert "concept_alignment_after" in data


def test_api_deliberation(client):
    res = client.post("/api/deliberation", json={"prompt": "Because reasoning matters", "max_tokens": 8})
    assert res.status_code == 200
    data = res.json()
    assert "total_tokens" in data
    assert "mean_entropy" in data
    assert "cognitive_certainty" in data
    assert "steps" in data
    assert len(data["steps"]) > 0


def test_api_sae(client):
    res = client.post("/api/sae/decompose", json={"prompt": "Nebula SAE", "layer_idx": 1, "k_sparsity": 8, "top_features": 4})
    assert res.status_code == 200
    data = res.json()
    assert "layer_idx" in data
    assert "l0_sparsity" in data
    assert "active_features" in data
    assert len(data["active_features"]) <= 4
    assert "reconstruction_loss" in data


def test_api_speculative_verification(client):
    res = client.post("/api/speculative/verify", json={"prefix": "Testing speculative", "draft_gamma": 3})
    assert res.status_code == 200
    data = res.json()
    assert "proposed_tokens" in data
    assert data["proposed_tokens"] == 3
    assert "accepted_tokens" in data
    assert "speedup_ratio" in data
    assert len(data["decisions"]) > 0


def test_api_induction_circuits(client):
    res = client.post("/api/circuits/induction", json={"seq_len": 16})
    assert res.status_code == 200
    data = res.json()
    assert "total_heads" in data
    assert "induction_heads" in data
    assert "top_induction_heads" in data
    assert len(data["top_induction_heads"]) > 0



