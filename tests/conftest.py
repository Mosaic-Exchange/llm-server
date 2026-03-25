import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests against a live server (requires servers on :4000 and :8080)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as requiring live servers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="pass --integration to run against live servers")
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip)
