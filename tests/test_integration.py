import pytest
import requests

MIDDLEWARE_URL = "http://127.0.0.1:4000"


# Session fixture — skip everything if the server isn't up
@pytest.fixture(scope="session")
def live_server():
    """Skip the entire suite if the middleware is not reachable."""
    try:
        r = requests.get(f"{MIDDLEWARE_URL}/health", timeout=3)
        r.raise_for_status()
    except Exception as e:
        pytest.skip(f"Middleware not reachable at {MIDDLEWARE_URL}: {e}")

# Helpers
def _assert_error_envelope(body, *, code=None):
    assert "error" in body
    err = body["error"]
    assert isinstance(err.get("message"), str) and err["message"]
    assert err.get("err_id", "").startswith("err_")
    if code is not None:
        assert err["code"] == code


# Health
@pytest.mark.integration
class TestIntegrationHealth:
    def test_middleware_ok(self, live_server):
        r = requests.get(f"{MIDDLEWARE_URL}/health")
        assert r.status_code == 200
        assert r.json()["middleware"] == "ok"

    def test_llama_server_reachable(self, live_server):
        r = requests.get(f"{MIDDLEWARE_URL}/health")
        assert r.json()["llama_server"] == "ok", (
            "llama.cpp server is not running, start it before running integration tests"
        )

    def test_adapters_registered_is_int(self, live_server):
        r = requests.get(f"{MIDDLEWARE_URL}/health")
        assert isinstance(r.json()["adapters_registered"], int)

# Adapters
@pytest.mark.integration
class TestIntegrationAdapters:
    def test_list_returns_list_object(self, live_server):
        r = requests.get(f"{MIDDLEWARE_URL}/v1/adapters")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)

    def test_delete_unknown_adapter_404(self, live_server):
        r = requests.delete(f"{MIDDLEWARE_URL}/v1/adapters/adp_doesnotexist")
        assert r.status_code == 404
        _assert_error_envelope(r.json(), code="ADAPTER_NOT_FOUND")

    def test_register_nonexistent_dir_404(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": "this_dir_does_not_exist_xyz", "request_id": "int-reg-404"},
        )
        assert r.status_code == 404
        _assert_error_envelope(r.json(), code="ADAPTER_DIR_NOT_FOUND")

# Generations
@pytest.mark.integration
class TestIntegrationGenerate:
    def test_generate_returns_response(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations",
            json={"message": "Say the word yes.", "request_id": "int-gen-001"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "generation"
        assert body["request_id"] == "int-gen-001"
        assert isinstance(body["output"], str) and len(body["output"]) > 0

    def test_generate_echoes_request_id(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations",
            json={"message": "Hello.", "request_id": "int-trace-abc"},
        )
        assert r.json()["request_id"] == "int-trace-abc"

    def test_generate_unknown_adapter_404(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations",
            json={
                "message": "Hello.",
                "request_id": "int-gen-adp-404",
                "adapter_id": "adp_doesnotexist",
            },
        )
        assert r.status_code == 404
        _assert_error_envelope(r.json(), code="ADAPTER_NOT_FOUND")

    def test_generate_max_tokens_out_of_range_400(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations",
            json={"message": "Hello.", "request_id": "int-gen-mt", "max_tokens": 0},
        )
        assert r.status_code == 400
        _assert_error_envelope(r.json(), code="INVALID_MAX_TOKENS")

    def test_generate_missing_fields_422(self, live_server):
        r = requests.post(f"{MIDDLEWARE_URL}/v1/generations", json={"message": "Hello."})
        assert r.status_code == 422