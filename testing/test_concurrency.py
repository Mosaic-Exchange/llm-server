import time
import uuid
import threading
import pytest
import requests
from concurrent.futures import ThreadPoolExecutor

MIDDLEWARE_URL = "http://127.0.0.1:4000"
MAX_CONCURRENT_GENERATIONS = 4
PROMPT = "Count from 1 to 5."
MAX_TOKENS = 80

CONCURRENT_LIMIT = 1.5  # concurrent wall time must be < CONCURRENT_LIMIT * single
SERIAL_LIMIT = 1.8      # serialized wall time must be >= SERIAL_LIMIT * single


@pytest.fixture(scope="session")
def live_server():
    try:
        r = requests.get(f"{MIDDLEWARE_URL}/health", timeout=3)
        r.raise_for_status()
    except Exception as e:
        pytest.skip(f"Middleware not reachable: {e}")


@pytest.fixture(scope="session")
def registered_adapters(live_server):
    r = requests.get(f"{MIDDLEWARE_URL}/v1/adapters", timeout=5)
    r.raise_for_status()
    return r.json().get("data", [])


def _stream_request(adapter_id=None):
    payload = {
        "message": PROMPT,
        "request_id": uuid.uuid4().hex[:8],
        "max_tokens": MAX_TOKENS,
    }
    if adapter_id:
        payload["adapter_id"] = adapter_id
    start = time.perf_counter()
    r = requests.post(
        f"{MIDDLEWARE_URL}/v1/generations/stream",
        json=payload,
        stream=True,
        timeout=120,
    )
    r.raise_for_status()
    output = "".join(chunk.decode("utf-8") for chunk in r.iter_content(chunk_size=None))
    return output, time.perf_counter() - start


def _run_concurrent(fns):
    """Fire all fns simultaneously from separate threads, return (results, wall_time)."""
    barrier = threading.Barrier(len(fns))

    def synchronized(fn):
        barrier.wait()
        return fn()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(fns)) as ex:
        futures = [ex.submit(synchronized, fn) for fn in fns]
        results = [f.result() for f in futures]
    return results, time.perf_counter() - start


def test_concurrent_base_model(live_server):
    """Two base-model requests should run concurrently (both take the read-path)."""
    # warm up, grab a single-request baseline
    _, single = _stream_request()
    _, wall = _run_concurrent([_stream_request, _stream_request])
    assert wall < single * CONCURRENT_LIMIT, (
        f"wall={wall:.2f}s exceeded {single * CONCURRENT_LIMIT:.2f}s limit "
        f"(single={single:.2f}s) - looks like requests serialized"
    )


def test_concurrent_same_adapter(live_server, registered_adapters):
    """Two requests for the same adapter should run concurrently after a warm-up."""
    if not registered_adapters:
        pytest.skip("no adapters registered")
    aid = registered_adapters[0]["adapter_id"]
    # warm up so the adapter is already loaded before the timed run
    _, single = _stream_request(adapter_id=aid)
    _, wall = _run_concurrent([
        lambda: _stream_request(adapter_id=aid),
        lambda: _stream_request(adapter_id=aid),
    ])
    assert wall < single * CONCURRENT_LIMIT, (
        f"wall={wall:.2f}s, limit={single * CONCURRENT_LIMIT:.2f}s "
        f"(baseline={single:.2f}s) - same-adapter requests may have serialized"
    )


def test_different_adapters_serialize(live_server, registered_adapters):
    """Requests for two different adapters must serialize - both need the write lock."""
    if len(registered_adapters) < 2:
        pytest.skip("need at least 2 adapters registered")
    aid_a = registered_adapters[0]["adapter_id"]
    aid_b = registered_adapters[1]["adapter_id"]
    # baseline with adapter A already loaded
    _, single = _stream_request(adapter_id=aid_a)
    _, wall = _run_concurrent([
        lambda: _stream_request(adapter_id=aid_a),
        lambda: _stream_request(adapter_id=aid_b),
    ])
    assert wall >= single * SERIAL_LIMIT, (
        f"wall={wall:.2f}s below {single * SERIAL_LIMIT:.2f}s threshold "
        f"(single={single:.2f}s) - adapters may not have serialized"
    )


def test_concurrency_cap(live_server):
    """(MAX_CONCURRENT_GENERATIONS+1)th request must queue behind the cap."""
    n = MAX_CONCURRENT_GENERATIONS
    _, wall_n = _run_concurrent([_stream_request] * n)
    _, wall_n_plus_1 = _run_concurrent([_stream_request] * (n + 1))
    # n+1 requests should take noticeably longer once one has to wait in queue
    assert wall_n_plus_1 > wall_n * 1.2, (
        f"wall_n={wall_n:.2f}s  wall_n+1={wall_n_plus_1:.2f}s - "
        f"request {n + 1} didn't appear to queue behind the cap of {n}"
    )