import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

_SETUP_DIR = str(Path(__file__).parent.parent / "setup")
if _SETUP_DIR not in sys.path:
    sys.path.insert(0, _SETUP_DIR)


def pytest_configure(config):
    """patch LlamaClient before middleware_server is imported during collection"""
    if "middleware_server" not in sys.modules:
        patch("LlamaClient.LlamaClient", autospec=True).start()


@pytest.fixture(scope="module")
def client():
    """configures mock defaults before startup event fires"""
    import middleware_server

    # _known_adapters is iterated in the startup event; must be an empty list.
    middleware_server._llama._known_adapters = []
    # Keep shutdown handler a no-op.
    middleware_server._llama._server_process = None
    # Safe defaults so tests that don't use mock_llama still get sensible behaviour.
    middleware_server._llama.chat.side_effect = None
    middleware_server._llama.chat.return_value = "mocked response"
    middleware_server._llama.get_system_prompt.return_value = None
    middleware_server._llama.get_parameter_suggestions.return_value = None

    with TestClient(middleware_server.app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_llama():
    """return the module-level _llama MagicMock with a clean slate for this test"""
    import middleware_server

    m = middleware_server._llama
    _methods = (
        m.chat,
        m.get_system_prompt,
        m.get_parameter_suggestions,
        m.convert_adapter,
        m.register_adapter,
    )
    for method in _methods:
        method.reset_mock()
        method.side_effect = None

    m.chat.return_value = "mocked response"
    m.get_system_prompt.return_value = None
    m.get_parameter_suggestions.return_value = None

    yield m
    for method in _methods:
        method.side_effect = None


@pytest.fixture(autouse=True)
def clean_adapters():
    """Snapshot and restore the adapter registry around every test """
    import middleware_server

    snapshot = dict(middleware_server._adapters)
    yield
    middleware_server._adapters.clear()
    middleware_server._adapters.update(snapshot)


@pytest.fixture
def tmp_adapters_dir(tmp_path, monkeypatch):
    """redirect LLAMA_ADAPTERS_DIR to a temporary directory for this test """
    import middleware_server

    monkeypatch.setattr(middleware_server, "LLAMA_ADAPTERS_DIR", str(tmp_path))
    return tmp_path