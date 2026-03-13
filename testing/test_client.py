"""
Direct llama.cpp server test client (port 8080).
Tests the llama.cpp server independently of the middleware.
"""

import json
import requests

LLAMA_URL = "http://127.0.0.1:8080"
PROMPT = "Tell me about the Eiffel Tower."


def check_health():
    print("=== Health Check ===")
    r = requests.get(f"{LLAMA_URL}/health")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    print()


def list_adapters():
    print("=== Loaded Adapters ===")
    r = requests.get(f"{LLAMA_URL}/lora-adapters")
    r.raise_for_status()
    adapters = r.json()
    print(json.dumps(adapters, indent=2))
    print()
    return adapters


def set_adapter_scales(scales: list):
    """scales: list of {"id": int, "scale": float}"""
    r = requests.post(f"{LLAMA_URL}/lora-adapters", json=scales)
    r.raise_for_status()
    return r.json()


def chat(message: str, label: str = ""):
    if label:
        print(f"=== {label} ===")
    payload = {
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.7,
        "max_tokens": 256,
    }
    r = requests.post(f"{LLAMA_URL}/v1/chat/completions", json=payload)
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]
    print(reply)
    print()
    return reply


if __name__ == "__main__":
    check_health()
    adapters = list_adapters()

    # --- Base model (all adapters at scale 0) ---
    if adapters:
        set_adapter_scales([{"id": a["id"], "scale": 0.0} for a in adapters])
    chat(PROMPT, label="Base Model")

    # --- Each adapter at full scale ---
    for adapter in adapters:
        set_adapter_scales([
            {"id": a["id"], "scale": 1.0 if a["id"] == adapter["id"] else 0.0}
            for a in adapters
        ])
        chat(PROMPT, label=f"Adapter {adapter['id']} (scale=1.0)")

    # --- Back to base ---
    if adapters:
        set_adapter_scales([{"id": a["id"], "scale": 0.0} for a in adapters])
    chat(PROMPT, label="Base Model (again)")
