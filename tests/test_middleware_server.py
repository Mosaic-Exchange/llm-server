"""
Unit tests for middleware_server.py.

Only tests that require mock injection are kept here:
  - Exception simulation (side_effect = RuntimeError)
  - Assertions on internal call arguments (what was forwarded to _llama)

Everything else lives in test_integration.py.
"""

import json
import uuid
from unittest.mock import patch

import pytest


def _add_adapter(adapters_dict, filename):
    adapter_id = "adp_" + uuid.uuid5(uuid.NAMESPACE_URL, filename).hex[:12]
    adapters_dict[adapter_id] = filename
    return adapter_id


def _make_adapter_dir(base, name, gguf=False, config=True, config_data=None, weights=True):
    d = base / name
    d.mkdir(exist_ok=True)
    if gguf:
        (d / f"{name}.gguf").touch()
    if config:
        data = config_data if config_data is not None else {"lora_alpha": 32, "r": 8}
        (d / "adapter_config.json").write_text(json.dumps(data))
    if weights:
        (d / "adapter_model.safetensors").touch()
    return d


# POST /v1/generations — parameter forwarding and exception injection
class TestGenerate:
    def test_generate_with_adapter_uses_correct_filename(self, client, mock_llama):
        import middleware_server

        filename = "mymodel/mymodel.gguf"
        aid = _add_adapter(middleware_server._adapters, filename)
        client.post(
            "/v1/generations",
            json={"message": "test", "request_id": "req-adp", "adapter_id": aid},
        )
        mock_llama.chat.assert_called_once()
        assert mock_llama.chat.call_args.kwargs.get("adapter_filename") == filename

    def test_generate_default_params_no_adapter(self, client, mock_llama):
        client.post("/v1/generations", json={"message": "test", "request_id": "req-def"})
        mock_llama.chat.assert_called_once()
        kwargs = mock_llama.chat.call_args.kwargs
        assert kwargs.get("temperature") == 0.7
        assert kwargs.get("max_tokens") == 256

    def test_generate_adapter_suggestions_temperature_applied(self, client, mock_llama):
        import middleware_server

        filename = "tempadp/tempadp.gguf"
        aid = _add_adapter(middleware_server._adapters, filename)
        mock_llama.get_parameter_suggestions.return_value = {"temperature": 1.5}
        client.post(
            "/v1/generations",
            json={"message": "test", "request_id": "req-temp", "adapter_id": aid},
        )
        assert mock_llama.chat.call_args.kwargs.get("temperature") == 1.5

    def test_generate_adapter_suggestions_min_p_applied(self, client, mock_llama):
        import middleware_server

        filename = "minadp/minadp.gguf"
        aid = _add_adapter(middleware_server._adapters, filename)
        mock_llama.get_parameter_suggestions.return_value = {"min_p": 0.1}
        client.post(
            "/v1/generations",
            json={"message": "test", "request_id": "req-minp", "adapter_id": aid},
        )
        assert mock_llama.chat.call_args.kwargs.get("min_p") == 0.1

    def test_generate_caller_max_tokens_overrides_suggestion(self, client, mock_llama):
        import middleware_server

        filename = "overrideadp/overrideadp.gguf"
        aid = _add_adapter(middleware_server._adapters, filename)
        mock_llama.get_parameter_suggestions.return_value = {"max_tokens": 128}
        client.post(
            "/v1/generations",
            json={
                "message": "test",
                "request_id": "req-ov",
                "adapter_id": aid,
                "max_tokens": 300,
            },
        )
        assert mock_llama.chat.call_args.kwargs.get("max_tokens") == 300

    def test_generate_system_prompt_passed_to_chat(self, client, mock_llama):
        import middleware_server

        filename = "sysadp/sysadp.gguf"
        aid = _add_adapter(middleware_server._adapters, filename)
        mock_llama.get_system_prompt.return_value = "Be Ella"
        client.post(
            "/v1/generations",
            json={"message": "test", "request_id": "req-sys", "adapter_id": aid},
        )
        assert mock_llama.chat.call_args.kwargs.get("system_prompt") == "Be Ella"

    def test_generate_no_system_prompt_passes_none(self, client, mock_llama):
        import middleware_server

        filename = "nosysadp/nosysadp.gguf"
        aid = _add_adapter(middleware_server._adapters, filename)
        mock_llama.get_system_prompt.return_value = None
        client.post(
            "/v1/generations",
            json={"message": "test", "request_id": "req-nosys", "adapter_id": aid},
        )
        assert mock_llama.chat.call_args.kwargs.get("system_prompt") is None

    def test_generate_backend_exception_500_with_message(self, client, mock_llama):
        mock_llama.chat.side_effect = RuntimeError("gpu error")
        r = client.post("/v1/generations", json={"message": "test", "request_id": "req-err"})
        assert r.status_code == 500
        err = r.json()["error"]
        assert err["code"] == "INTERNAL_ERROR"
        assert "gpu error" in err["message"]


# POST /v1/adapters — conversion flow and internal call assertions
class TestCreateAdapter:
    def test_create_conversion_called_with_correct_path(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "convtest")
        gguf_result = tmp_adapters_dir / "convtest" / "convtest.gguf"
        mock_llama.convert_adapter.return_value = gguf_result
        client.post("/v1/adapters", json={"adapter_dir": "convtest", "request_id": "req-conv"})
        mock_llama.convert_adapter.assert_called_once()
        assert mock_llama.convert_adapter.call_args.args[0].name == "convtest"

    def test_create_conversion_success_registers_adapter(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "regconv")
        gguf_result = tmp_adapters_dir / "regconv" / "regconv.gguf"
        mock_llama.convert_adapter.return_value = gguf_result
        client.post("/v1/adapters", json={"adapter_dir": "regconv", "request_id": "req-rc"})
        mock_llama.register_adapter.assert_called_once()
        assert mock_llama.register_adapter.call_args.args[0] == "regconv/regconv.gguf"

    def test_create_conversion_failure_500(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "failconv")
        mock_llama.convert_adapter.side_effect = RuntimeError("bad weights")
        r = client.post("/v1/adapters", json={"adapter_dir": "failconv", "request_id": "req-fail"})
        assert r.status_code == 500
        err = r.json()["error"]
        assert err["code"] == "CONVERSION_FAILED"
        assert "bad weights" in err["message"]

    def test_create_weight_file_deleted_after_conversion(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "delweights")
        weight_file = tmp_adapters_dir / "delweights" / "adapter_model.safetensors"
        assert weight_file.exists()
        gguf_result = tmp_adapters_dir / "delweights" / "delweights.gguf"
        mock_llama.convert_adapter.return_value = gguf_result
        client.post("/v1/adapters", json={"adapter_dir": "delweights", "request_id": "req-delw"})
        assert not weight_file.exists()

    def test_create_system_prompt_passed_to_register(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "sysmodel", gguf=True, config=False, weights=False)
        (tmp_adapters_dir / "sysmodel" / "system_prompt.txt").write_text("Hello\n")
        client.post("/v1/adapters", json={"adapter_dir": "sysmodel", "request_id": "req-sys"})
        assert mock_llama.register_adapter.call_args.kwargs.get("system_prompt") == "Hello"

    def test_create_no_system_prompt_passes_none(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "nosysmodel", gguf=True, config=False, weights=False)
        client.post("/v1/adapters", json={"adapter_dir": "nosysmodel", "request_id": "req-nosys"})
        assert mock_llama.register_adapter.call_args.kwargs.get("system_prompt") is None

    def test_create_suggestions_allowed_keys_passed(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "sugmodel", gguf=True, config=False, weights=False)
        suggestions = {"temperature": 0.9, "min_p": 0.05, "max_tokens": 64}
        (tmp_adapters_dir / "sugmodel" / "parameter_suggestions.json").write_text(
            json.dumps(suggestions)
        )
        client.post("/v1/adapters", json={"adapter_dir": "sugmodel", "request_id": "req-sug"})
        assert mock_llama.register_adapter.call_args.kwargs.get("parameter_suggestions") == suggestions

    def test_create_suggestions_unknown_keys_stripped(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "stripmodel", gguf=True, config=False, weights=False)
        (tmp_adapters_dir / "stripmodel" / "parameter_suggestions.json").write_text(
            json.dumps({"temperature": 0.9, "foo": 99})
        )
        client.post("/v1/adapters", json={"adapter_dir": "stripmodel", "request_id": "req-strip"})
        assert mock_llama.register_adapter.call_args.kwargs.get("parameter_suggestions") == {
            "temperature": 0.9
        }

    def test_create_malformed_suggestions_json_ignored(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "badjson", gguf=True, config=False, weights=False)
        (tmp_adapters_dir / "badjson" / "parameter_suggestions.json").write_text("not valid json{{")
        client.post("/v1/adapters", json={"adapter_dir": "badjson", "request_id": "req-bj"})
        assert mock_llama.register_adapter.call_args.kwargs.get("parameter_suggestions") is None


# GET /health — exception injection to simulate unreachable llama server
class TestHealth:
    def test_health_llama_unreachable(self, client):
        with patch("middleware_server.requests.get", side_effect=ConnectionError("refused")):
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["llama_server"] == "unreachable"
