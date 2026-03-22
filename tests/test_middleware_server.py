import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest



def _add_adapter(adapters_dict, filename):
    """Directly register an adapter in the in-memory dict. Returns adapter_id"""
    adapter_id = "adp_" + uuid.uuid5(uuid.NAMESPACE_URL, filename).hex[:12]
    adapters_dict[adapter_id] = filename
    return adapter_id


def _make_adapter_dir(base, name, gguf=False, config=True, config_data=None, weights=True):
    """Create a minimal adapter directory structure under base/name"""
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


def _assert_error_envelope(body, *, err_type=None, code=None):
    """Assert the standard error envelope shape"""
    assert "error" in body
    err = body["error"]
    assert isinstance(err.get("message"), str) and err["message"]
    assert err.get("err_id", "").startswith("err_")
    if err_type is not None:
        assert err["type"] == err_type
    if code is not None:
        assert err["code"] == code


# GET /v1/adapters
class TestListAdapters:
    def test_list_empty_returns_list_object(self, client):
        r = client.get("/v1/adapters")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert r.json() == {"object": "list", "data": []}

    def test_list_returns_correct_adapter_shape(self, client):
        import middleware_server

        _add_adapter(middleware_server._adapters, "a/a.gguf")
        r = client.get("/v1/adapters")
        body = r.json()
        assert body["object"] == "list"
        item = body["data"][0]
        assert "adapter_id" in item
        assert "adapter_filename" in item
        assert item["object"] == "adapter"

    def test_list_multiple_adapters_all_present(self, client):
        import middleware_server

        ids = [
            _add_adapter(middleware_server._adapters, f"adp{i}/adp{i}.gguf")
            for i in range(3)
        ]
        r = client.get("/v1/adapters")
        data = r.json()["data"]
        assert len(data) == 3
        returned_ids = {item["adapter_id"] for item in data}
        for aid in ids:
            assert aid in returned_ids


# DELETE /v1/adapters/{adapter_id}
class TestDeleteAdapter:
    def test_delete_returns_full_adapter_object(self, client):
        import middleware_server

        aid = _add_adapter(middleware_server._adapters, "del/del.gguf")
        r = client.delete(f"/v1/adapters/{aid}")
        assert r.status_code == 200
        body = r.json()
        assert body["adapter_id"] == aid
        assert body["adapter_filename"] == "del/del.gguf"
        assert body["object"] == "adapter"

    def test_delete_removes_from_registry(self, client):
        import middleware_server

        aid = _add_adapter(middleware_server._adapters, "gone/gone.gguf")
        client.delete(f"/v1/adapters/{aid}")
        ids = {item["adapter_id"] for item in client.get("/v1/adapters").json()["data"]}
        assert aid not in ids

    def test_delete_only_removes_target(self, client):
        import middleware_server

        aid1 = _add_adapter(middleware_server._adapters, "keep/keep.gguf")
        aid2 = _add_adapter(middleware_server._adapters, "drop/drop.gguf")
        client.delete(f"/v1/adapters/{aid2}")
        ids = {item["adapter_id"] for item in client.get("/v1/adapters").json()["data"]}
        assert aid1 in ids
        assert aid2 not in ids

    def test_delete_not_found_error_envelope(self, client):
        r = client.delete("/v1/adapters/adp_doesnotexist")
        assert r.status_code == 404
        _assert_error_envelope(r.json(), err_type="not_found_error", code="ADAPTER_NOT_FOUND")


# POST /v1/generations
class TestGenerate:
    def test_generate_success_full_schema(self, client):
        r = client.post("/v1/generations", json={"message": "hello", "request_id": "req-001"})
        assert r.status_code == 200
        body = r.json()
        assert "request_id" in body
        assert body["object"] == "generation"
        assert isinstance(body["output"], str) and body["output"]

    def test_generate_echoes_request_id(self, client):
        r = client.post("/v1/generations", json={"message": "hi", "request_id": "my-trace-id"})
        assert r.json()["request_id"] == "my-trace-id"

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

    def test_generate_unknown_adapter_id_error_envelope(self, client):
        r = client.post(
            "/v1/generations",
            json={"message": "hi", "request_id": "req-unk", "adapter_id": "adp_doesnotexist"},
        )
        assert r.status_code == 404
        _assert_error_envelope(r.json(), err_type="not_found_error", code="ADAPTER_NOT_FOUND")

    def test_generate_unknown_adapter_err_id_contains_request_id(self, client):
        r = client.post(
            "/v1/generations",
            json={"message": "hi", "request_id": "req-xyz", "adapter_id": "adp_doesnotexist"},
        )
        assert r.json()["error"]["err_id"] == "err_req-xyz"

    def test_generate_max_tokens_0_rejected(self, client):
        r = client.post(
            "/v1/generations", json={"message": "hi", "request_id": "req-mt0", "max_tokens": 0}
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_MAX_TOKENS"

    def test_generate_max_tokens_501_rejected(self, client):
        r = client.post(
            "/v1/generations", json={"message": "hi", "request_id": "req-mt501", "max_tokens": 501}
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_MAX_TOKENS"

    def test_generate_max_tokens_boundary_1_accepted(self, client):
        r = client.post(
            "/v1/generations", json={"message": "hi", "request_id": "req-mt1", "max_tokens": 1}
        )
        assert r.status_code == 200

    def test_generate_max_tokens_boundary_500_accepted(self, client):
        r = client.post(
            "/v1/generations", json={"message": "hi", "request_id": "req-mt500", "max_tokens": 500}
        )
        assert r.status_code == 200

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

    def test_generate_missing_message_field_422(self, client):
        r = client.post("/v1/generations", json={"request_id": "req-nm"})
        assert r.status_code == 422

    def test_generate_missing_request_id_field_422(self, client):
        r = client.post("/v1/generations", json={"message": "hi"})
        assert r.status_code == 422


# POST /v1/adapters
class TestCreateAdapter:
    def test_create_dir_not_found(self, client, tmp_adapters_dir):
        r = client.post(
            "/v1/adapters", json={"adapter_dir": "nonexistent_xyz", "request_id": "req-dnf"}
        )
        assert r.status_code == 404
        err = r.json()["error"]
        assert err["code"] == "ADAPTER_DIR_NOT_FOUND"
        assert "nonexistent_xyz" in err["message"]

    def test_create_gguf_already_exists_skips_conversion(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "mymodel", gguf=True, config=False, weights=False)
        r = client.post(
            "/v1/adapters", json={"adapter_dir": "mymodel", "request_id": "req-gguf"}
        )
        assert r.status_code == 200
        mock_llama.convert_adapter.assert_not_called()
        body = r.json()
        assert "adapter_id" in body
        assert "adapter_filename" in body
        assert body["object"] == "adapter"

    def test_create_gguf_already_exists_appears_in_registry(
        self, client, tmp_adapters_dir, mock_llama
    ):
        _make_adapter_dir(tmp_adapters_dir, "regtest", gguf=True, config=False, weights=False)
        r = client.post(
            "/v1/adapters", json={"adapter_dir": "regtest", "request_id": "req-reg"}
        )
        aid = r.json()["adapter_id"]
        listed = {item["adapter_id"] for item in client.get("/v1/adapters").json()["data"]}
        assert aid in listed

    def test_create_missing_config_json(self, client, tmp_adapters_dir):
        d = tmp_adapters_dir / "noconfig"
        d.mkdir()
        (d / "adapter_model.safetensors").touch()
        r = client.post(
            "/v1/adapters", json={"adapter_dir": "noconfig", "request_id": "req-nc"}
        )
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "MISSING_ADAPTER_FILES"
        assert "adapter_config.json" in err["message"]

    def test_create_missing_weight_files(self, client, tmp_adapters_dir):
        d = tmp_adapters_dir / "noweights"
        d.mkdir()
        (d / "adapter_config.json").write_text('{"lora_alpha": 32}')
        r = client.post(
            "/v1/adapters", json={"adapter_dir": "noweights", "request_id": "req-nw"}
        )
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "MISSING_ADAPTER_FILES"
        assert "adapter_model" in err["message"]

    def test_create_missing_both_files_mentions_both(self, client, tmp_adapters_dir):
        (tmp_adapters_dir / "noboth").mkdir()
        r = client.post(
            "/v1/adapters", json={"adapter_dir": "noboth", "request_id": "req-nb"}
        )
        assert r.status_code == 400
        msg = r.json()["error"]["message"]
        assert "adapter_config.json" in msg
        assert "adapter_model" in msg

    def test_create_incompatible_format_no_lora_alpha(self, client, tmp_adapters_dir):
        d = tmp_adapters_dir / "mlxfmt"
        d.mkdir()
        (d / "adapter_config.json").write_text('{"r": 8}')
        (d / "adapter_model.safetensors").touch()
        r = client.post(
            "/v1/adapters", json={"adapter_dir": "mlxfmt", "request_id": "req-incompat"}
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INCOMPATIBLE_ADAPTER_FORMAT"

    def test_create_conversion_called_with_correct_path(
        self, client, tmp_adapters_dir, mock_llama
    ):
        _make_adapter_dir(tmp_adapters_dir, "convtest")
        gguf_result = tmp_adapters_dir / "convtest" / "convtest.gguf"
        mock_llama.convert_adapter.return_value = gguf_result
        client.post("/v1/adapters", json={"adapter_dir": "convtest", "request_id": "req-conv"})
        mock_llama.convert_adapter.assert_called_once()
        called_path = mock_llama.convert_adapter.call_args.args[0]
        assert called_path.name == "convtest"

    def test_create_conversion_success_registers_adapter(
        self, client, tmp_adapters_dir, mock_llama
    ):
        _make_adapter_dir(tmp_adapters_dir, "regconv")
        gguf_result = tmp_adapters_dir / "regconv" / "regconv.gguf"
        mock_llama.convert_adapter.return_value = gguf_result
        client.post("/v1/adapters", json={"adapter_dir": "regconv", "request_id": "req-rc"})
        mock_llama.register_adapter.assert_called_once()
        registered_filename = mock_llama.register_adapter.call_args.args[0]
        assert registered_filename == "regconv/regconv.gguf"

    def test_create_conversion_success_response_schema(
        self, client, tmp_adapters_dir, mock_llama
    ):
        _make_adapter_dir(tmp_adapters_dir, "schematest")
        gguf_result = tmp_adapters_dir / "schematest" / "schematest.gguf"
        mock_llama.convert_adapter.return_value = gguf_result
        r = client.post(
            "/v1/adapters", json={"adapter_dir": "schematest", "request_id": "req-schema"}
        )
        assert r.status_code == 200
        body = r.json()
        assert "adapter_id" in body
        assert "adapter_filename" in body
        assert body["object"] == "adapter"

    def test_create_conversion_failure_500(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "failconv")
        mock_llama.convert_adapter.side_effect = RuntimeError("bad weights")
        r = client.post(
            "/v1/adapters", json={"adapter_dir": "failconv", "request_id": "req-fail"}
        )
        assert r.status_code == 500
        err = r.json()["error"]
        assert err["code"] == "CONVERSION_FAILED"
        assert "bad weights" in err["message"]

    def test_create_adapter_id_is_deterministic(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "dettest", gguf=True, config=False, weights=False)
        r1 = client.post(
            "/v1/adapters", json={"adapter_dir": "dettest", "request_id": "req-det1"}
        )
        r2 = client.post(
            "/v1/adapters", json={"adapter_dir": "dettest", "request_id": "req-det2"}
        )
        assert r1.json()["adapter_id"] == r2.json()["adapter_id"]

    def test_create_system_prompt_passed_to_register(
        self, client, tmp_adapters_dir, mock_llama
    ):
        _make_adapter_dir(tmp_adapters_dir, "sysmodel", gguf=True, config=False, weights=False)
        (tmp_adapters_dir / "sysmodel" / "system_prompt.txt").write_text("Hello\n")
        client.post("/v1/adapters", json={"adapter_dir": "sysmodel", "request_id": "req-sys"})
        call_kwargs = mock_llama.register_adapter.call_args.kwargs
        assert call_kwargs.get("system_prompt") == "Hello"

    def test_create_no_system_prompt_passes_none(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "nosysmodel", gguf=True, config=False, weights=False)
        client.post(
            "/v1/adapters", json={"adapter_dir": "nosysmodel", "request_id": "req-nosys"}
        )
        call_kwargs = mock_llama.register_adapter.call_args.kwargs
        assert call_kwargs.get("system_prompt") is None

    def test_create_suggestions_allowed_keys_passed(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "sugmodel", gguf=True, config=False, weights=False)
        suggestions = {"temperature": 0.9, "min_p": 0.05, "max_tokens": 64}
        (tmp_adapters_dir / "sugmodel" / "parameter_suggestions.json").write_text(
            json.dumps(suggestions)
        )
        client.post("/v1/adapters", json={"adapter_dir": "sugmodel", "request_id": "req-sug"})
        call_kwargs = mock_llama.register_adapter.call_args.kwargs
        assert call_kwargs.get("parameter_suggestions") == suggestions

    def test_create_suggestions_unknown_keys_stripped(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "stripmodel", gguf=True, config=False, weights=False)
        (tmp_adapters_dir / "stripmodel" / "parameter_suggestions.json").write_text(
            json.dumps({"temperature": 0.9, "foo": 99})
        )
        client.post(
            "/v1/adapters", json={"adapter_dir": "stripmodel", "request_id": "req-strip"}
        )
        call_kwargs = mock_llama.register_adapter.call_args.kwargs
        assert call_kwargs.get("parameter_suggestions") == {"temperature": 0.9}

    def test_create_malformed_suggestions_json_ignored(
        self, client, tmp_adapters_dir, mock_llama
    ):
        _make_adapter_dir(tmp_adapters_dir, "badjson", gguf=True, config=False, weights=False)
        (tmp_adapters_dir / "badjson" / "parameter_suggestions.json").write_text(
            "not valid json{{"
        )
        client.post("/v1/adapters", json={"adapter_dir": "badjson", "request_id": "req-bj"})
        call_kwargs = mock_llama.register_adapter.call_args.kwargs
        assert call_kwargs.get("parameter_suggestions") is None

    def test_create_weight_file_deleted_after_conversion(
        self, client, tmp_adapters_dir, mock_llama
    ):
        _make_adapter_dir(tmp_adapters_dir, "delweights")
        weight_file = tmp_adapters_dir / "delweights" / "adapter_model.safetensors"
        assert weight_file.exists()
        gguf_result = tmp_adapters_dir / "delweights" / "delweights.gguf"
        mock_llama.convert_adapter.return_value = gguf_result
        client.post(
            "/v1/adapters", json={"adapter_dir": "delweights", "request_id": "req-delw"}
        )
        assert not weight_file.exists()


# GET /health
class TestHealth:
    def test_health_middleware_always_ok(self, client):
        with patch("middleware_server.requests.get", side_effect=ConnectionError("refused")):
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["middleware"] == "ok"

    def test_health_llama_reachable(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}
        with patch("middleware_server.requests.get", return_value=mock_resp):
            r = client.get("/health")
        assert r.json()["llama_server"] == "ok"

    def test_health_llama_unreachable(self, client):
        with patch("middleware_server.requests.get", side_effect=ConnectionError("refused")):
            r = client.get("/health")
        assert r.json()["llama_server"] == "unreachable"

    def test_health_adapters_registered_count(self, client):
        import middleware_server

        _add_adapter(middleware_server._adapters, "a1/a1.gguf")
        _add_adapter(middleware_server._adapters, "a2/a2.gguf")
        with patch("middleware_server.requests.get", side_effect=ConnectionError):
            r = client.get("/health")
        assert r.json()["adapters_registered"] == 2


# End-to-end flow tests (stateful within each test)
class TestFlows:
    def test_flow_register_list_delete_list(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "flowadp", gguf=True, config=False, weights=False)

        # Register
        r = client.post(
            "/v1/adapters", json={"adapter_dir": "flowadp", "request_id": "req-flow1"}
        )
        assert r.status_code == 200
        aid = r.json()["adapter_id"]

        #verify listed
        listed = {item["adapter_id"] for item in client.get("/v1/adapters").json()["data"]}
        assert aid in listed

        #delete
        r = client.delete(f"/v1/adapters/{aid}")
        assert r.status_code == 200

        # verify gone
        listed = {item["adapter_id"] for item in client.get("/v1/adapters").json()["data"]}
        assert aid not in listed

    def test_flow_register_then_generate(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "genflow", gguf=True, config=False, weights=False)

        r = client.post(
            "/v1/adapters", json={"adapter_dir": "genflow", "request_id": "req-gf1"}
        )
        assert r.status_code == 200
        aid = r.json()["adapter_id"]
        filename = r.json()["adapter_filename"]

        mock_llama.chat.reset_mock()
        r = client.post(
            "/v1/generations",
            json={"message": "hello", "request_id": "req-gf2", "adapter_id": aid},
        )
        assert r.status_code == 200
        assert mock_llama.chat.call_args.kwargs.get("adapter_filename") == filename

    def test_flow_delete_then_generate_fails(self, client, tmp_adapters_dir, mock_llama):
        _make_adapter_dir(tmp_adapters_dir, "delgen", gguf=True, config=False, weights=False)

        r = client.post(
            "/v1/adapters", json={"adapter_dir": "delgen", "request_id": "req-dg1"}
        )
        aid = r.json()["adapter_id"]

        client.delete(f"/v1/adapters/{aid}")

        r = client.post(
            "/v1/generations",
            json={"message": "hello", "request_id": "req-dg2", "adapter_id": aid},
        )
        assert r.status_code == 404