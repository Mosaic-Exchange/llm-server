import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "setup"))

from fastapi.testclient import TestClient
from MiddleManAPILogic import app

client = TestClient(app)

def test_create_generation_no_adapter():
    response = client.post("/generation", json={
        "message": "What is the speed of light?",
        "request_id": "req-001"
    })
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-001"
    assert "output" in body

def test_create_generation_with_adapter():
    response = client.post("/generation", json={
        "message": "Explain quantum entanglement.",
        "request_id": "req-002",
        "adapter_id": "physics_expert"
    })
    assert response.status_code == 200
    assert response.json()["request_id"] == "req-002"

def test_create_adapter():
    response = client.post("/adapter", json={
        "adapter_filename": "physics_expert.gguf",
        "request_id": "req-003"
    })
    assert response.status_code == 200
    body = response.json()
    assert body["adapter_filename"] == "physics_expert.gguf"
    assert "adapter_id" in body

def test_delete_adapter():
    response = client.request("DELETE", "/adapter", json={
        "adapter_id": "stub-adapter-id-001"
    })
    assert response.status_code == 200
    assert response.json()["deleted"] is True

def test_list_adapters():
    response = client.get("/adapters")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
