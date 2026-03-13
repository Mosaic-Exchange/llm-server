"""
Interactive chat client for the mosaicAI middleware (port 4000).

Each round shows:
- All registered adapters (available to select)
- Which adapters are hot-loaded in the llama server, and what % of the
  capacity (MAX_LOADED = 2) that represents
Then prompts for an adapter choice and a message.
"""

import sys
import uuid
import requests

MIDDLEWARE_URL = "http://127.0.0.1:4000"
LLAMA_URL      = "http://127.0.0.1:8080"
MAX_LOADED     = 2  # LlamaClient.MAX_LOADED_ADAPTERS


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_registered_adapters() -> list[dict]:
    """GET /v1/adapters → list of {adapter_id, adapter_filename}"""
    r = requests.get(f"{MIDDLEWARE_URL}/v1/adapters", timeout=5)
    r.raise_for_status()
    return r.json().get("data", [])


def get_active_adapters() -> list[dict]:
    """GET llama /lora-adapters → list of currently hot-loaded adapter objects."""
    try:
        r = requests.get(f"{LLAMA_URL}/lora-adapters", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def generate(message: str, adapter_id: str | None) -> str:
    payload = {
        "message": message,
        "request_id": uuid.uuid4().hex[:8],
    }
    if adapter_id:
        payload["adapter_id"] = adapter_id

    r = requests.post(f"{MIDDLEWARE_URL}/v1/generations", json=payload, timeout=120)
    if r.status_code != 200:
        err = r.json().get("error", {})
        return f"[ERROR {r.status_code}] {err.get('message', r.text)}"
    return r.json().get("output", "")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def short_name(filename: str) -> str:
    """Strip directory prefix and .gguf suffix for a readable label."""
    name = filename.split("/")[-1]
    if name.endswith(".gguf"):
        name = name[:-5]
    return name


def print_adapter_table(registered: list[dict], active: list[dict]) -> None:
    # Build a set of filenames (or path fragments) that are hot-loaded
    active_paths = {a.get("path", "") for a in active}

    active_count = len(active)
    pct = int(active_count / MAX_LOADED * 100)
    print(f"\n  Hot-loaded: {active_count}/{MAX_LOADED} ({pct}% capacity)\n")

    print(f"  {'#':>2}  {'Adapter':<35}  {'Status'}")
    print(f"  {'--':>2}  {'-'*35}  {'------'}")
    print(f"  {'0':>2}  {'(no adapter)':<35}")

    for i, adp in enumerate(registered, start=1):
        name  = short_name(adp["adapter_filename"])
        label = f"{name:<35}"
        # Mark as active if any hot-loaded path contains this adapter's filename
        is_active = any(adp["adapter_filename"] in p or name in p for p in active_paths)
        status = "● loaded" if is_active else ""
        print(f"  {i:>2}  {label}  {status}")

    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("mosaicAI interactive chat  (Ctrl-C or 'quit' to exit)\n")

    while True:
        # ---- fetch state ----
        try:
            registered = get_registered_adapters()
        except requests.exceptions.ConnectionError:
            print("[ERROR] Cannot reach middleware at", MIDDLEWARE_URL)
            sys.exit(1)

        active = get_active_adapters()

        # ---- display ----
        print_adapter_table(registered, active)

        # ---- adapter selection ----
        while True:
            raw = input("  Select adapter (0 = none): ").strip()
            if raw.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                sys.exit(0)
            if raw.isdigit() and 0 <= int(raw) <= len(registered):
                choice = int(raw)
                break
            print(f"  Please enter a number between 0 and {len(registered)}.")

        selected_id   = registered[choice - 1]["adapter_id"] if choice > 0 else None
        selected_name = short_name(registered[choice - 1]["adapter_filename"]) if choice > 0 else "base model"

        # ---- query ----
        query = input(f"\n  [{selected_name}] You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            sys.exit(0)
        if not query:
            continue

        # ---- generate ----
        print("\n  Generating...", end="\r")
        response = generate(query, selected_id)
        print(f"  Assistant: {response}\n")
        print("  " + "-" * 60)


if __name__ == "__main__":
    main()
