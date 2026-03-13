"""
Timing comparison: direct llama.cpp (port 8080) vs middleware (port 4000).
Measures per-run times for base model and each adapter, then summarises overhead.
"""

import time
import requests

LLAMA_URL = "http://127.0.0.1:8080"
MIDDLEWARE_URL = "http://127.0.0.1:4000"
PROMPT = "Tell me about the Eiffel Tower."
RUNS = 3  # runs per condition


# ------------------------------------------------------------------
# Direct llama.cpp helpers
# ------------------------------------------------------------------

def llama_get_adapters() -> list:
    r = requests.get(f"{LLAMA_URL}/lora-adapters")
    r.raise_for_status()
    return r.json()


def llama_set_scales(adapters: list, active_id: int | None):
    scales = [
        {"id": a["id"], "scale": 1.0 if a["id"] == active_id else 0.0}
        for a in adapters
    ]
    requests.post(f"{LLAMA_URL}/lora-adapters", json=scales).raise_for_status()


def llama_chat_timed(message: str) -> tuple[str, float]:
    payload = {
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.7,
        "max_tokens": 256,
    }
    start = time.perf_counter()
    r = requests.post(f"{LLAMA_URL}/v1/chat/completions", json=payload)
    elapsed = time.perf_counter() - start
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], elapsed


# ------------------------------------------------------------------
# Middleware helpers
# ------------------------------------------------------------------

def middleware_get_adapters() -> list:
    """Returns list of {adapter_id, adapter_filename} from the middleware."""
    r = requests.get(f"{MIDDLEWARE_URL}/v1/adapters")
    r.raise_for_status()
    return r.json()["data"]


def middleware_chat_timed(message: str, adapter_id: str | None = None) -> tuple[str, float]:
    payload = {
        "message": message,
        "request_id": f"timing-{time.time_ns()}",
        "max_tokens": 256,
    }
    if adapter_id:
        payload["adapter_id"] = adapter_id

    start = time.perf_counter()
    r = requests.post(f"{MIDDLEWARE_URL}/v1/generations", json=payload)
    elapsed = time.perf_counter() - start
    r.raise_for_status()
    return r.json()["output"], elapsed


# ------------------------------------------------------------------
# Generic run helper
# ------------------------------------------------------------------

def run_timed(label: str, fn, runs: int = RUNS) -> list[float]:
    """Call fn() `runs` times, print per-run times, return list of elapsed times."""
    print(f"\n  {label}")
    times = []
    last_reply = ""
    for i in range(1, runs + 1):
        reply, elapsed = fn()
        times.append(elapsed)
        last_reply = reply
        print(f"    Run {i}: {elapsed:.3f}s")
    avg = sum(times) / len(times)
    print(f"    Avg: {avg:.3f}s  Min: {min(times):.3f}s  Max: {max(times):.3f}s")
    print(f"    Response: {last_reply[:200]}{'...' if len(last_reply) > 200 else ''}")
    return times


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    llama_adapters = llama_get_adapters()
    middleware_adapters = middleware_get_adapters()

    print(f"llama.cpp adapters loaded : {[a['id'] for a in llama_adapters]}")
    print(f"Middleware adapters known : {[a['adapter_filename'] for a in middleware_adapters]}")

    results: dict[str, dict[str, list[float]]] = {}

    # ---- Direct: base model ----
    section = "Base model"
    results[section] = {}
    print(f"\n{'='*60}\n  DIRECT  —  {section}\n{'='*60}")
    llama_set_scales(llama_adapters, active_id=None)
    results[section]["direct"] = run_timed(
        "Direct to llama.cpp",
        lambda: llama_chat_timed(PROMPT),
    )

    # ---- Middleware: base model ----
    print(f"\n{'='*60}\n  MIDDLEWARE  —  {section}\n{'='*60}")
    results[section]["middleware"] = run_timed(
        "Through middleware (no adapter_id)",
        lambda: middleware_chat_timed(PROMPT),
    )

    # ---- Per-adapter ----
    for m_adapter in middleware_adapters:
        filename = m_adapter["adapter_filename"]
        adapter_id = m_adapter["adapter_id"]

        # Find the matching llama.cpp integer id by filename
        llama_id = next(
            (a["id"] for a in llama_adapters if filename in str(a.get("path", ""))),
            llama_adapters[0]["id"] if llama_adapters else None,
        )

        section = f"Adapter: {filename}"
        results[section] = {}

        print(f"\n{'='*60}\n  DIRECT  —  {section}\n{'='*60}")
        if llama_id is not None:
            llama_set_scales(llama_adapters, active_id=llama_id)
        results[section]["direct"] = run_timed(
            f"Direct to llama.cpp (id={llama_id})",
            lambda: llama_chat_timed(PROMPT),
        )

        print(f"\n{'='*60}\n  MIDDLEWARE  —  {section}\n{'='*60}")
        results[section]["middleware"] = run_timed(
            f"Through middleware (adapter_id={adapter_id})",
            lambda aid=adapter_id: middleware_chat_timed(PROMPT, adapter_id=aid),
        )

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({RUNS} runs each)")
    print(f"{'='*60}")
    print(f"  {'Condition':<35} {'Direct':>8} {'Middleware':>12} {'Overhead':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*12} {'-'*10}")
    for section, data in results.items():
        d_avg = sum(data["direct"]) / len(data["direct"])
        m_avg = sum(data["middleware"]) / len(data["middleware"])
        overhead = m_avg - d_avg
        sign = "+" if overhead >= 0 else ""
        print(f"  {section:<35} {d_avg:>7.3f}s {m_avg:>11.3f}s {sign}{overhead:>8.3f}s")
