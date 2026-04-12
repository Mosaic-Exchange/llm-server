import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from starlette.testclient import TestClient

_SETUP_DIR = str(Path(__file__).parent.parent / "setup")
if _SETUP_DIR not in sys.path:
    sys.path.insert(0, _SETUP_DIR)


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests against the live servers running on 4000 and 8080",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as requiring live servers")
    if "middleware_server" not in sys.modules: #if server not running
        patch("LlamaClient.LlamaClient", autospec=True).start()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="pass integration to run against live servers")
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip)


@pytest.fixture(scope="module")
def client(): #make mock defaults
    import middleware_server
    middleware_server._llama._known_adapters = [] #_known_adapters is iterated at startup
    middleware_server._llama._server_process = None
    middleware_server._llama.chat.side_effect = None
    middleware_server._llama.chat.return_value = "mocked response"
    middleware_server._llama.get_system_prompt.return_value = None
    middleware_server._llama.get_parameter_suggestions.return_value = None

    with TestClient(middleware_server.app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_llama():
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
def clean_adapters(): #restore adapter registry at every run
    import middleware_server
    snapshot = dict(middleware_server._adapters)
    yield
    middleware_server._adapters.clear()
    middleware_server._adapters.update(snapshot)


@pytest.fixture
def tmp_adapters_dir(tmp_path, monkeypatch): # redirect to temp dir
    import middleware_server

    monkeypatch.setattr(middleware_server, "LLAMA_ADAPTERS_DIR", str(tmp_path))
    return tmp_path