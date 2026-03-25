# llama_client.py
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

ADAPTER_MEMORY_PATH = Path(__file__).parent.parent / "database" / "adaper_memory.json"


class LlamaClient:
    """Client for interacting with llama.cpp server with hot-swappable LoRA adapters"""

    MAX_LOADED_ADAPTERS = 2

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        server_binary: Optional[str] = None,
        model_path: Optional[str] = None,
        adapters_dir: Optional[str] = None,
        convert_script: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._server_binary = server_binary
        self._model_path = model_path
        self._adapters_dir = Path(adapters_dir) if adapters_dir else None
        self._convert_script = Path(convert_script) if convert_script else None
        self._server_process: Optional[subprocess.Popen] = None
        self._port = urlparse(self.base_url).port or 8080

        # All ever-registered adapter filenames
        self._known_adapters: List[str] = []
        # Currently loaded adapters, ordered LRU→MRU (index 0 = LRU, last = MRU)
        self._active_adapters: List[str] = []
        # Optional system prompts keyed by adapter filename
        self._system_prompts: Dict[str, str] = {}
        # Optional generation parameter suggestions keyed by adapter filename
        self._parameter_suggestions: Dict[str, Dict[str, Any]] = {}

        self._load_adapter_memory()
        self._ensure_server_running()

    # ------------------------------------------------------------------
    # Adapter memory (persistence)
    # ------------------------------------------------------------------

    def _load_adapter_memory(self):
        """Load adapter lists from persistent storage on startup."""
        if not ADAPTER_MEMORY_PATH.exists():
            return
        with open(ADAPTER_MEMORY_PATH, "r") as f:
            data = json.load(f)
        # Migrate old format: {"adapters": [{"filename": ..., "id": ...}]}
        if "adapters" in data and "known_adapters" not in data:
            filenames = [a["filename"] for a in data["adapters"]]
            self._known_adapters = filenames
            self._active_adapters = filenames[:]
        else:
            self._known_adapters = data.get("known_adapters", [])
            self._active_adapters = data.get("active_adapters", [])
            self._system_prompts = data.get("system_prompts", {})
            self._parameter_suggestions = data.get("parameter_suggestions", {})

    def _save_adapter_memory(self):
        """Persist adapter lists to disk."""
        ADAPTER_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ADAPTER_MEMORY_PATH, "w") as f:
            json.dump(
                {
                    "known_adapters": self._known_adapters,
                    "active_adapters": self._active_adapters,
                    "system_prompts": self._system_prompts,
                    "parameter_suggestions": self._parameter_suggestions,
                },
                f,
                indent=2,
            )

    def register_adapter(
        self,
        filename: str,
        system_prompt: Optional[str] = None,
        parameter_suggestions: Optional[Dict[str, Any]] = None,
    ):
        """Add an adapter to the known list. Optionally stores a system prompt and parameter suggestions."""
        if filename not in self._known_adapters:
            self._known_adapters.append(filename)
        if system_prompt is not None:
            self._system_prompts[filename] = system_prompt
        if parameter_suggestions is not None:
            self._parameter_suggestions[filename] = parameter_suggestions
        self._save_adapter_memory()

    def get_system_prompt(self, filename: str) -> Optional[str]:
        """Return the system prompt for an adapter, or None if not set."""
        return self._system_prompts.get(filename)

    def get_parameter_suggestions(self, filename: str) -> Optional[Dict[str, Any]]:
        """Return generation parameter suggestions for an adapter, or None if not set."""
        return self._parameter_suggestions.get(filename)

    def convert_adapter(self, adapter_dir: Path) -> Path:
        """
        Run convert_lora_to_gguf.py on adapter_dir and return the path to the
        resulting GGUF file. Raises RuntimeError if conversion fails.
        """
        convert_script = self._convert_script
        base_config = Path(__file__).parent / "llama32_3b_config"
        outfile = adapter_dir / f"{adapter_dir.name}.gguf"

        cmd = [
            sys.executable,
            str(convert_script),
            "--base", str(base_config),
            "--outfile", str(outfile),
            "--outtype", "q8_0",
            str(adapter_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)

        return outfile

    # ------------------------------------------------------------------
    # LRU management + server reload
    # ------------------------------------------------------------------

    def _ensure_adapter_active(self, filename: str) -> int:
        """
        Ensure the adapter is in the active set, evicting LRU and reloading
        the server if necessary. Returns the llama.cpp integer id (index in
        active_adapters after the update).
        """
        if filename not in self._known_adapters:
            raise ValueError(f"Adapter '{filename}' is not in the known adapters list.")

        if filename in self._active_adapters:
            # Already loaded — just update LRU order, no reload needed
            self._active_adapters.remove(filename)
            self._active_adapters.append(filename)
            self._save_adapter_memory()
        else:
            # Not currently loaded — evict LRU if at capacity, then reload
            if len(self._active_adapters) >= self.MAX_LOADED_ADAPTERS:
                evicted = self._active_adapters.pop(0)
                logger.info(f"Evicting LRU adapter: {evicted}")
            self._active_adapters.append(filename)
            self._save_adapter_memory()
            self._reload_server()

        return self._active_adapters.index(filename)

    def _ensure_server_running(self):
        """Start the llama-server if it is not already responding."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2)
            if r.status_code == 200:
                return  # already up
        except Exception:
            pass

        if not self._server_binary or not self._model_path:
            raise RuntimeError(
                "llama-server is not running and server_binary/model_path were not provided to start it."
            )

        logger.info("llama-server not detected — starting it now.")
        self._reload_server()

    def _reload_server(self):
        """Kill the current llama-server and restart it with the active adapter set."""
        if not self._server_binary or not self._model_path:
            raise RuntimeError(
                "server_binary and model_path must be provided to LlamaClient to support adapter reload."
            )

        # Terminate any managed server process
        if self._server_process and self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()

        cmd = [
            self._server_binary,
            "-m", self._model_path,
            "--host", "127.0.0.1",
            "--port", str(self._port),
        ]
        for filename in self._active_adapters:
            adapter_path = (self._adapters_dir / filename) if self._adapters_dir else Path(filename)
            cmd.extend(["--lora", str(adapter_path)])

        logger.info(f"Reloading llama-server: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 120):
        """Poll the health endpoint until the server is ready."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    logger.info("llama-server is ready.")
                    return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError(f"llama-server did not become ready within {timeout}s.")

    # ------------------------------------------------------------------
    # Adapter control (llama.cpp API)
    # ------------------------------------------------------------------

    def list_adapters(self) -> List[Dict]:
        """List all loaded LoRA adapters from the llama.cpp server."""
        response = requests.get(f"{self.base_url}/lora-adapters")
        response.raise_for_status()
        return response.json()

    def set_adapters(self, adapters: List[int]) -> Dict:
        """Set which adapters to use for inference."""
        response = requests.post(f"{self.base_url}/lora-adapters", json=adapters)
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            response.raise_for_status()
        return response.json()

    def use_base_only(self) -> Dict:
        """Switch to base model (set all adapter scales to 0)."""
        adapters_info = self.list_adapters()
        payload = [{"id": a["id"], "scale": 0.0} for a in adapters_info]
        response = requests.post(f"{self.base_url}/lora-adapters", json=payload)
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            response.raise_for_status()
        return response.json()

    def use_adapter(self, adapter_id: int, scale: float = 1.0) -> Dict:
        """Activate a specific adapter by llama.cpp integer id at the given scale."""
        adapters_info = self.list_adapters()
        payload = [
            {"id": a["id"], "scale": scale if a["id"] == adapter_id else 0.0}
            for a in adapters_info
        ]
        response = requests.post(f"{self.base_url}/lora-adapters", json=payload)
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            response.raise_for_status()
        return response.json()

    def use_adapter_by_name(self, filename: str, scale: float = 1.0) -> int:
        """
        Ensure the adapter is active (evicting LRU + reloading if needed),
        then activate it at the given scale. Returns the llama.cpp integer id.
        """
        llama_id = self._ensure_adapter_active(filename)
        self.use_adapter(llama_id, scale)
        return llama_id

    def set_adapter_scales(self, scales: Dict[int, float]) -> Dict:
        """Set scales for multiple adapters at once."""
        adapters_info = self.list_adapters()
        payload = [{"id": a["id"], "scale": scales.get(a["id"], 0.0)} for a in adapters_info]
        response = requests.post(f"{self.base_url}/lora-adapters", json=payload)
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Chat / completions
    # ------------------------------------------------------------------

    def chat(
        self,
        message: Union[str, List[Dict[str, str]]],
        adapter_id: Optional[int] = None,
        adapter_filename: Optional[str] = None,
        adapter_scale: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 500,
        min_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> Union[str, iter]:
        """
        Send a chat completion request.

        Prefer adapter_filename — it supports LRU eviction and automatic reload.
        adapter_id (integer) is kept for backward compatibility but bypasses LRU logic.
        """
        if adapter_filename is not None:
            self.use_adapter_by_name(adapter_filename, adapter_scale)
        elif adapter_id is not None:
            self.use_adapter(adapter_id, adapter_scale)

        if isinstance(message, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message})
        else:
            messages = message

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if min_p is not None:
            payload["min_p"] = min_p

        if stream:
            return self._stream_chat(payload)
        else:
            return self._send_chat(payload)

    def _send_chat(self, payload: Dict) -> str:
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _stream_chat(self, payload: Dict):
        payload["stream"] = True
        response = requests.post(
            f"{self.base_url}/v1/chat/completions", json=payload, stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get("choices"):
                            content = chunk["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

    def completion(
        self,
        prompt: str,
        adapter_id: Optional[int] = None,
        adapter_filename: Optional[str] = None,
        adapter_scale: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 500,
        stop: Optional[List[str]] = None,
    ) -> str:
        if adapter_filename is not None:
            self.use_adapter_by_name(adapter_filename, adapter_scale)
        elif adapter_id is not None:
            self.use_adapter(adapter_id, adapter_scale)

        payload = {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
        if stop:
            payload["stop"] = stop

        response = requests.post(f"{self.base_url}/v1/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]


if __name__ == "__main__":
    client = LlamaClient()

    print("=== Checking loaded adapters ===")
    adapters = client.list_adapters()
    print(json.dumps(adapters, indent=2))
    print()

    print("=== Test 1: Base Model ===")
    client.use_base_only()
    response = client.chat("Say 'I am the base model'")
    print(response)
    print()

    print("=== Test 2: Adapter by name ===")
    response = client.chat(
        "Say 'I am using the adapter'",
        adapter_filename="structured_answer_adapter.gguf",
    )
    print(response)
