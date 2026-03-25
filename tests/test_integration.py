"""
Integration tests — require live middleware (port 4000) and llama.cpp (port 8080).

Run with:
    .mosaic-venv/bin/python -m pytest tests/test_integration.py --integration -v
"""

import json
import shutil
import uuid
from pathlib import Path

import pytest
import requests

MIDDLEWARE_URL = "http://127.0.0.1:4000"
ADAPTERS_DIR = Path(__file__).parent.parent / "adapters"

# Fixtures
@pytest.fixture(scope="session")
def live_server():
    """Skip the entire suite if the middleware is not reachable."""
    try:
        r = requests.get(f"{MIDDLEWARE_URL}/health", timeout=3)
        r.raise_for_status()
    except Exception as e:
        pytest.skip(f"Middleware not reachable at {MIDDLEWARE_URL}: {e}")

@pytest.fixture
def tmp_adapter_dir():
    """Create a uniquely-named subdirectory in the real adapters/ folder, remove after test."""
    name = f"_test_{uuid.uuid4().hex[:8]}"
    d = ADAPTERS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    yield name, d
    if d.exists():
        shutil.rmtree(d)


@pytest.fixture
def registered_adapter(live_server, tmp_adapter_dir):
    """Register a fake (empty .gguf) adapter against the live server. Unregisters after test."""
    name, d = tmp_adapter_dir
    (d / f"{name}.gguf").touch()
    r = requests.post(
        f"{MIDDLEWARE_URL}/v1/adapters",
        json={"adapter_dir": name, "request_id": "fixture-register"},
    )
    assert r.status_code == 200, f"Fixture setup failed: {r.json()}"
    data = r.json()
    yield data
    requests.delete(f"{MIDDLEWARE_URL}/v1/adapters/{data['adapter_id']}")

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
            "llama.cpp server is not running — start it before running integration tests"
        )

    def test_adapters_registered_is_int(self, live_server):
        r = requests.get(f"{MIDDLEWARE_URL}/health")
        assert isinstance(r.json()["adapters_registered"], int)

    def test_adapters_registered_count_reflects_registry(self, live_server, registered_adapter):
        before = requests.get(f"{MIDDLEWARE_URL}/health").json()["adapters_registered"]
        assert before >= 1


# Adapters — list
@pytest.mark.integration
class TestIntegrationListAdapters:
    def test_list_returns_list_object(self, live_server):
        r = requests.get(f"{MIDDLEWARE_URL}/v1/adapters")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)

    def test_list_registered_adapter_has_correct_shape(self, live_server, registered_adapter):
        r = requests.get(f"{MIDDLEWARE_URL}/v1/adapters")
        found = next(
            (item for item in r.json()["data"] if item["adapter_id"] == registered_adapter["adapter_id"]),
            None,
        )
        assert found is not None
        assert "adapter_id" in found
        assert "adapter_filename" in found
        assert found["object"] == "adapter"

    def test_list_registered_adapter_appears(self, live_server, registered_adapter):
        ids = {item["adapter_id"] for item in requests.get(f"{MIDDLEWARE_URL}/v1/adapters").json()["data"]}
        assert registered_adapter["adapter_id"] in ids

# Adapters — delete
@pytest.mark.integration
class TestIntegrationDeleteAdapter:
    def test_delete_returns_adapter_object(self, live_server, tmp_adapter_dir):
        name, d = tmp_adapter_dir
        (d / f"{name}.gguf").touch()
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": name, "request_id": "req-del-setup"},
        )
        assert r.status_code == 200
        aid, filename = r.json()["adapter_id"], r.json()["adapter_filename"]

        r = requests.delete(f"{MIDDLEWARE_URL}/v1/adapters/{aid}")
        assert r.status_code == 200
        body = r.json()
        assert body["adapter_id"] == aid
        assert body["adapter_filename"] == filename
        assert body["object"] == "adapter"

    def test_delete_removes_from_registry(self, live_server, registered_adapter):
        aid = registered_adapter["adapter_id"]
        requests.delete(f"{MIDDLEWARE_URL}/v1/adapters/{aid}")
        ids = {item["adapter_id"] for item in requests.get(f"{MIDDLEWARE_URL}/v1/adapters").json()["data"]}
        assert aid not in ids

    def test_delete_only_removes_target(self, live_server):
        # Create and register two independent adapters
        dirs = []
        aids = []
        for _ in range(2):
            name = f"_test_{uuid.uuid4().hex[:8]}"
            d = ADAPTERS_DIR / name
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.gguf").touch()
            dirs.append(d)
            r = requests.post(
                f"{MIDDLEWARE_URL}/v1/adapters",
                json={"adapter_dir": name, "request_id": f"req-{name}"},
            )
            assert r.status_code == 200
            aids.append(r.json()["adapter_id"])

        try:
            requests.delete(f"{MIDDLEWARE_URL}/v1/adapters/{aids[1]}")
            ids = {item["adapter_id"] for item in requests.get(f"{MIDDLEWARE_URL}/v1/adapters").json()["data"]}
            assert aids[0] in ids
            assert aids[1] not in ids
        finally:
            requests.delete(f"{MIDDLEWARE_URL}/v1/adapters/{aids[0]}")
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_delete_unknown_adapter_404(self, live_server):
        r = requests.delete(f"{MIDDLEWARE_URL}/v1/adapters/adp_doesnotexist")
        assert r.status_code == 404
        _assert_error_envelope(r.json(), code="ADAPTER_NOT_FOUND")

# Adapters — create
@pytest.mark.integration
class TestIntegrationCreateAdapter:
    def test_register_nonexistent_dir_404(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": "this_dir_does_not_exist_xyz", "request_id": "int-reg-404"},
        )
        assert r.status_code == 404
        _assert_error_envelope(r.json(), code="ADAPTER_DIR_NOT_FOUND")

    def test_register_missing_config_json_400(self, live_server, tmp_adapter_dir):
        name, d = tmp_adapter_dir
        (d / "adapter_model.safetensors").touch()
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": name, "request_id": "req-nc"},
        )
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "MISSING_ADAPTER_FILES"
        assert "adapter_config.json" in err["message"]

    def test_register_missing_weights_400(self, live_server, tmp_adapter_dir):
        name, d = tmp_adapter_dir
        (d / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32}))
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": name, "request_id": "req-nw"},
        )
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "MISSING_ADAPTER_FILES"
        assert "adapter_model" in err["message"]

    def test_register_missing_both_mentions_both(self, live_server, tmp_adapter_dir):
        name, d = tmp_adapter_dir
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": name, "request_id": "req-nb"},
        )
        assert r.status_code == 400
        msg = r.json()["error"]["message"]
        assert "adapter_config.json" in msg
        assert "adapter_model" in msg

    def test_register_incompatible_format_no_lora_alpha(self, live_server, tmp_adapter_dir):
        name, d = tmp_adapter_dir
        (d / "adapter_config.json").write_text(json.dumps({"r": 8}))
        (d / "adapter_model.safetensors").touch()
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": name, "request_id": "req-incompat"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INCOMPATIBLE_ADAPTER_FORMAT"

    def test_register_gguf_exists_returns_adapter_object(self, live_server, registered_adapter):
        assert "adapter_id" in registered_adapter
        assert "adapter_filename" in registered_adapter
        assert registered_adapter["object"] == "adapter"

    def test_register_adapter_id_is_deterministic(self, live_server, tmp_adapter_dir):
        name, d = tmp_adapter_dir
        (d / f"{name}.gguf").touch()
        r1 = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": name, "request_id": "req-det1"},
        )
        r2 = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": name, "request_id": "req-det2"},
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["adapter_id"] == r2.json()["adapter_id"]
        requests.delete(f"{MIDDLEWARE_URL}/v1/adapters/{r1.json()['adapter_id']}")

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
            json={"message": "Hello.", "request_id": "int-gen-adp-404", "adapter_id": "adp_doesnotexist"},
        )
        assert r.status_code == 404
        _assert_error_envelope(r.json(), code="ADAPTER_NOT_FOUND")

    def test_generate_unknown_adapter_err_id_contains_request_id(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations",
            json={"message": "hi", "request_id": "req-xyz", "adapter_id": "adp_doesnotexist"},
        )
        assert r.json()["error"]["err_id"] == "err_req-xyz"

    def test_generate_max_tokens_0_rejected(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations",
            json={"message": "Hello.", "request_id": "int-gen-mt0", "max_tokens": 0},
        )
        assert r.status_code == 400
        _assert_error_envelope(r.json(), code="INVALID_MAX_TOKENS")

    def test_generate_max_tokens_501_rejected(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations",
            json={"message": "Hello.", "request_id": "int-gen-mt501", "max_tokens": 501},
        )
        assert r.status_code == 400
        _assert_error_envelope(r.json(), code="INVALID_MAX_TOKENS")

    def test_generate_max_tokens_boundary_1_accepted(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations",
            json={"message": "Hello.", "request_id": "int-gen-mt1", "max_tokens": 1},
        )
        assert r.status_code == 200

    def test_generate_max_tokens_boundary_500_accepted(self, live_server):
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations",
            json={"message": "Hello.", "request_id": "int-gen-mt500", "max_tokens": 500},
        )
        assert r.status_code == 200

    def test_generate_missing_message_field_422(self, live_server):
        r = requests.post(f"{MIDDLEWARE_URL}/v1/generations", json={"request_id": "req-nm"})
        assert r.status_code == 422

    def test_generate_missing_request_id_field_422(self, live_server):
        r = requests.post(f"{MIDDLEWARE_URL}/v1/generations", json={"message": "hi"})
        assert r.status_code == 422

# End-to-end flows
@pytest.mark.integration
class TestIntegrationFlows:
    def test_flow_register_list_delete_list(self, live_server, tmp_adapter_dir):
        name, d = tmp_adapter_dir
        (d / f"{name}.gguf").touch()

        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": name, "request_id": "req-flow1"},
        )
        assert r.status_code == 200
        aid = r.json()["adapter_id"]

        ids = {item["adapter_id"] for item in requests.get(f"{MIDDLEWARE_URL}/v1/adapters").json()["data"]}
        assert aid in ids

        requests.delete(f"{MIDDLEWARE_URL}/v1/adapters/{aid}")

        ids = {item["adapter_id"] for item in requests.get(f"{MIDDLEWARE_URL}/v1/adapters").json()["data"]}
        assert aid not in ids

    def test_flow_delete_then_generate_fails(self, live_server, tmp_adapter_dir):
        name, d = tmp_adapter_dir
        (d / f"{name}.gguf").touch()

        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/adapters",
            json={"adapter_dir": name, "request_id": "req-dg1"},
        )
        assert r.status_code == 200
        aid = r.json()["adapter_id"]

        requests.delete(f"{MIDDLEWARE_URL}/v1/adapters/{aid}")

        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations",
            json={"message": "hello", "request_id": "req-dg2", "adapter_id": aid},
        )
        assert r.status_code == 404