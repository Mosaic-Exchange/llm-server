from pathlib import Path
import pytest
from middleware_server import _err_id_from_request_id, _validate_adapter_dir

# _err_id_from_request_id
class TestErrIdFromRequestId:
    def test_with_request_id_preserves_value(self):
        assert _err_id_from_request_id("req-123") == "err_req-123"

    def test_with_none_starts_with_err(self):
        result = _err_id_from_request_id(None)
        assert result.startswith("err_")
        assert len(result) > 4  # has a UUID suffix

    def test_two_none_calls_unique(self):
        assert _err_id_from_request_id(None) != _err_id_from_request_id(None)

    def test_with_empty_string_uses_uuid_branch(self):
        # Empty string is falsy → UUID branch
        result = _err_id_from_request_id("")
        assert result.startswith("err_")
        assert len(result) > 4

    def test_deterministic_for_same_input(self):
        assert _err_id_from_request_id("abc") == _err_id_from_request_id("abc")

# _validate_adapter_dir
class TestValidateAdapterDir:
    def test_nonexistent_path_returns_none(self, tmp_path):
        assert _validate_adapter_dir(tmp_path / "nonexistent") is None

    def test_file_not_dir_returns_none(self, tmp_path):
        f = tmp_path / "file.txt"
        f.touch()
        assert _validate_adapter_dir(f) is None

    def test_valid_with_safetensors_returns_none(self, tmp_path):
        d = tmp_path / "adapter"
        d.mkdir()
        (d / "adapter_config.json").touch()
        (d / "adapter_model.safetensors").touch()
        assert _validate_adapter_dir(d) is None

    def test_valid_with_bin_returns_none(self, tmp_path):
        d = tmp_path / "adapter"
        d.mkdir()
        (d / "adapter_config.json").touch()
        (d / "adapter_model.bin").touch()
        assert _validate_adapter_dir(d) is None

    def test_missing_config_json_mentions_filename(self, tmp_path):
        d = tmp_path / "adapter"
        d.mkdir()
        (d / "adapter_model.safetensors").touch()
        result = _validate_adapter_dir(d)
        assert result is not None
        assert "adapter_config.json" in result

    def test_missing_weights_mentions_weights(self, tmp_path):
        d = tmp_path / "adapter"
        d.mkdir()
        (d / "adapter_config.json").touch()
        result = _validate_adapter_dir(d)
        assert result is not None
        assert "adapter_model" in result

    def test_missing_both_mentions_both(self, tmp_path):
        d = tmp_path / "adapter"
        d.mkdir()
        result = _validate_adapter_dir(d)
        assert result is not None
        assert "adapter_config.json" in result
        assert "adapter_model" in result

    def test_never_raises_always_returns_string_or_none(self, tmp_path):
        candidates = [
            tmp_path / "x",
            tmp_path,
            Path("/nonexistent/path/xyz"),
        ]
        for p in candidates:
            try:
                result = _validate_adapter_dir(p)
                assert result is None or isinstance(result, str)
            except Exception as exc:
                pytest.fail(f"_validate_adapter_dir raised {exc!r} for {p}")