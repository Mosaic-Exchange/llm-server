"""
mosaicAI Middleware Server (FastAPI)

Implements the API described in "AI Team Design Doc.pdf":
- POST   /v1/generations
- POST   /v1/adapters
- DELETE /v1/adapters/{adapter_id}
- GET    /v1/adapters

Notes:
- No database is used; adapter registry is in-memory (per the doc).
- Adapter selection on the underlying llama.cpp server is global state, so /v1/generations
  uses an asyncio lock to prevent cross-request adapter switching races.

Run:
  pip install fastapi uvicorn pydantic requests
  python -m uvicorn middleware_server:app --host 127.0.0.1 --port 4000
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Literal, Optional

import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import your llama client. Ensure LlamaClient.py (or llama_client.py) is on PYTHONPATH.
from LlamaClient import LlamaClient  # uploaded file name


# ----------------------------
# Config
# ----------------------------
LLAMA_BASE_URL = "http://127.0.0.1:8080"
_PROJECT_ROOT = Path(__file__).parent.parent
LLAMA_SERVER_BINARY = str(_PROJECT_ROOT / "llama.cpp" / "llama-server")
LLAMA_MODEL_PATH = str(_PROJECT_ROOT / "llama.cpp" / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf")
LLAMA_ADAPTERS_DIR = str(_PROJECT_ROOT / "adapters")
LLAMA_CONVERT_SCRIPT = str(_PROJECT_ROOT / "utilities" / "convert_lora_to_gguf.py")
MAX_TOKENS_MIN = 1
MAX_TOKENS_MAX = 500  # per API doc
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7

# ----------------------------
# App + shared state
# ----------------------------
app = FastAPI(title="mosaicAI Middleware", version="1.0")

_llama = LlamaClient(
    base_url=LLAMA_BASE_URL,
    server_binary=LLAMA_SERVER_BINARY,
    model_path=LLAMA_MODEL_PATH,
    adapters_dir=LLAMA_ADAPTERS_DIR,
    convert_script=LLAMA_CONVERT_SCRIPT,
)

# Adapter registry (in-memory): adapter_id -> adapter_filename
# The design doc says no DB; this resets when the server restarts.
_adapters: Dict[str, str] = {}

# Critical: llama.cpp adapter selection is process-global; serialize generation calls.
_generation_lock = asyncio.Lock()


@app.on_event("startup")
async def _autoload_adapters_from_memory():
    """Populate the adapter registry from adaper_memory.json on startup."""
    for filename in _llama._known_adapters:
        # Deterministic adapter_id based on filename so IDs are stable across restarts
        adapter_id = "adp_" + uuid.uuid5(uuid.NAMESPACE_URL, filename).hex[:12]
        _adapters[adapter_id] = filename


@app.on_event("shutdown")
async def _shutdown():
    """Terminate the managed llama-server process on shutdown."""
    import subprocess
    if _llama._server_process and _llama._server_process.poll() is None:
        _llama._server_process.terminate()
        try:
            _llama._server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _llama._server_process.kill()


# ----------------------------
# API models
# ----------------------------
class ErrorInner(BaseModel):
    type: Literal["invalid_request_error", "not_found_error", "api_error"]
    code: str
    message: str
    err_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorInner


class GenerationCreateRequest(BaseModel):
    message: str = Field(..., description="The input message for the LLM to generate a response to")
    request_id: str = Field(..., description="Caller-provided identifier for tracing")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    adapter_id: Optional[str] = Field(None, description="Identifier of the adapter to apply for this generation")


class GenerationObject(BaseModel):
    request_id: str
    object: Literal["generation"] = "generation"
    output: str


class AdapterCreateRequest(BaseModel):
    adapter_dir: str = Field(..., description="Name of the adapter subdirectory inside the adapters folder")
    request_id: str = Field(..., description="Caller-provided identifier for tracing")


class AdapterObject(BaseModel):
    adapter_id: str
    adapter_filename: str
    object: Literal["adapter"] = "adapter"


class AdapterListObject(BaseModel):
    object: Literal["list"] = "list"
    data: List[AdapterObject]


# ----------------------------
# Error helpers
# ----------------------------
def _err_id_from_request_id(request_id: Optional[str]) -> str:
    # Doc says err_id is unique based on request id for tracing.
    if request_id:
        # Preserve the caller's id inside the err_id for easy correlation.
        return f"err_{request_id}"
    return f"err_{uuid.uuid4().hex[:10]}"


def _error_response(
    *,
    http_status: int,
    err_type: Literal["invalid_request_error", "not_found_error", "api_error"],
    code: str,
    message: str,
    request_id: Optional[str],
) -> JSONResponse:
    body = {"error": {"type": err_type, "code": code, "message": message, "err_id": _err_id_from_request_id(request_id)}}
    return JSONResponse(status_code=http_status, content=body)


# Ensure FastAPI validation errors match the envelope (best-effort)
@app.exception_handler(HTTPException)
async def _http_exception_handler(_request: Request, exc: HTTPException):
    # If the handler is called with our own JSONResponse, it won't reach here.
    # For other HTTPExceptions, wrap into the standard envelope.
    return _error_response(
        http_status=int(exc.status_code),
        err_type="invalid_request_error" if 400 <= exc.status_code < 500 else "api_error",
        code="INVALID_REQUEST",
        message=str(exc.detail),
        request_id=None,
    )


# ----------------------------
# Adapter utilities
# ----------------------------
_REQUIRED_ADAPTER_FILES = {"adapter_config.json"}
_ADAPTER_WEIGHTS_OPTIONS = {"adapter_model.safetensors", "adapter_model.bin"}

def _validate_adapter_dir(adapter_path: Path) -> Optional[str]:
    if not adapter_path.exists() or not adapter_path.is_dir():
        return None  # caller handles the 404 case separately
    missing = []
    if not (adapter_path / "adapter_config.json").exists():
        missing.append("adapter_config.json")
    if not any((adapter_path / f).exists() for f in _ADAPTER_WEIGHTS_OPTIONS):
        missing.append("adapter_model.safetensors (or adapter_model.bin)")
    return f"Missing required files: {', '.join(missing)}" if missing else None


def _adapter_exists_in_registry(adapter_id: str) -> bool:
    return adapter_id in _adapters


def _resolve_adapter_for_llama(adapter_id: str) -> Dict[str, Any]:
    """Translate a middleware adapter_id (UUID) to the filename the LlamaClient expects."""
    if adapter_id in _adapters:
        return {"adapter_filename": _adapters[adapter_id]}
    raise KeyError("unmappable_adapter_id")


# ----------------------------
# Endpoints: Generations
# ----------------------------
@app.post("/v1/generations", response_model=GenerationObject, responses={400: {"model": ErrorEnvelope}, 404: {"model": ErrorEnvelope}, 500: {"model": ErrorEnvelope}})
async def create_generation(req: GenerationCreateRequest):
    # Validate adapter_id if provided
    adapter_kwargs: Dict[str, Any] = {}
    if req.adapter_id:
        if not _adapter_exists_in_registry(req.adapter_id):
            return _error_response(
                http_status=404,
                err_type="not_found_error",
                code="ADAPTER_NOT_FOUND",
                message="The specified adapter_id does not exist.",
                request_id=req.request_id,
            )
        try:
            adapter_kwargs = _resolve_adapter_for_llama(req.adapter_id)
        except KeyError:
            return _error_response(
                http_status=404,
                err_type="not_found_error",
                code="ADAPTER_NOT_FOUND",
                message="The specified adapter_id does not exist (cannot be mapped for backend).",
                request_id=req.request_id,
            )

    # Resolve generation parameters: system defaults → adapter suggestions → caller-provided
    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS
    min_p = None
    system_prompt = None

    if req.adapter_id and req.adapter_id in _adapters:
        filename = _adapters[req.adapter_id]
        system_prompt = _llama.get_system_prompt(filename)
        suggestions = _llama.get_parameter_suggestions(filename)
        if suggestions:
            temperature = suggestions.get("temperature", temperature)
            max_tokens = suggestions.get("max_tokens", max_tokens)
            min_p = suggestions.get("min_p", min_p)

    # Caller-provided max_tokens wins over adapter suggestion
    if req.max_tokens is not None:
        max_tokens = req.max_tokens

    if not (MAX_TOKENS_MIN <= max_tokens <= MAX_TOKENS_MAX):
        return _error_response(
            http_status=400,
            err_type="invalid_request_error",
            code="INVALID_MAX_TOKENS",
            message=f"max tokens must be between {MAX_TOKENS_MIN} and {MAX_TOKENS_MAX}.",
            request_id=req.request_id,
        )

    # Call llama backend in a critical section (adapter switching is global state)
    async with _generation_lock:
        try:
            output = _llama.chat(
                message=req.message,
                temperature=temperature,
                max_tokens=max_tokens,
                min_p=min_p,
                stream=False,
                system_prompt=system_prompt,
                **adapter_kwargs,
            )
        except Exception as e:
            return _error_response(
                http_status=500,
                err_type="api_error",
                code="INTERNAL_ERROR",
                message=f"An unexpected error occurred on the backend server: {e}",
                request_id=req.request_id,
            )

    return GenerationObject(request_id=req.request_id, output=str(output))


# ----------------------------
# Endpoints: Adapters
# ----------------------------
@app.post("/v1/adapters", response_model=AdapterObject, responses={400: {"model": ErrorEnvelope}, 404: {"model": ErrorEnvelope}, 500: {"model": ErrorEnvelope}})
async def create_adapter(req: AdapterCreateRequest):
    import json as _json
    import logging

    adapter_path = Path(LLAMA_ADAPTERS_DIR) / req.adapter_dir

    # Check directory exists
    if not adapter_path.exists() or not adapter_path.is_dir():
        return _error_response(
            http_status=404,
            err_type="not_found_error",
            code="ADAPTER_DIR_NOT_FOUND",
            message=f"Directory '{req.adapter_dir}' was not found in the adapters folder.",
            request_id=req.request_id,
        )

    gguf_path = Path(LLAMA_ADAPTERS_DIR) / req.adapter_dir / f"{req.adapter_dir}.gguf"

    if not gguf_path.exists():
        # Reject non-PEFT adapters before checking weight files (gives a clearer error)
        config_path = adapter_path / "adapter_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    _cfg = _json.load(f)
                if "lora_alpha" not in _cfg:
                    return _error_response(
                        http_status=400,
                        err_type="invalid_request_error",
                        code="INCOMPATIBLE_ADAPTER_FORMAT",
                        message="Only HuggingFace PEFT adapters are supported. This adapter appears to use a different format (e.g. MLX) and cannot be converted.",
                        request_id=req.request_id,
                    )
            except ValueError:
                pass  # malformed JSON is caught later by the conversion step

        # Check required files are present
        msg = _validate_adapter_dir(adapter_path)
        if msg:
            return _error_response(
                http_status=400,
                err_type="invalid_request_error",
                code="MISSING_ADAPTER_FILES",
                message=msg,
                request_id=req.request_id,
            )

        loop = asyncio.get_event_loop()
        try:
            gguf_path = await loop.run_in_executor(None, _llama.convert_adapter, adapter_path)
        except RuntimeError as e:
            return _error_response(
                http_status=500,
                err_type="api_error",
                code="CONVERSION_FAILED",
                message=str(e),
                request_id=req.request_id,
            )

        # Delete weight files — no longer needed once GGUF exists
        for disposable in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"):
            p = adapter_path / disposable
            if p.exists():
                try:
                    p.unlink()
                except OSError as e:
                    logging.warning("Could not delete %s: %s", p, e)

    gguf_filename = str(gguf_path.relative_to(Path(LLAMA_ADAPTERS_DIR)))
    adapter_id = "adp_" + uuid.uuid5(uuid.NAMESPACE_URL, gguf_filename).hex[:12]
    _adapters[adapter_id] = gguf_filename

    prompt_file = adapter_path / "system_prompt.txt"
    system_prompt = prompt_file.read_text().strip() if prompt_file.exists() else None

    suggestions_file = adapter_path / "parameter_suggestions.json"
    parameter_suggestions = None
    if suggestions_file.exists():
        try:
            raw = _json.loads(suggestions_file.read_text())
            _ALLOWED_PARAMS = {"temperature", "min_p", "max_tokens"}
            parameter_suggestions = {k: v for k, v in raw.items() if k in _ALLOWED_PARAMS}
        except ValueError:
            pass  # malformed JSON — silently ignore, adapter still registers without suggestions

    _llama.register_adapter(gguf_filename, system_prompt=system_prompt, parameter_suggestions=parameter_suggestions)

    return AdapterObject(adapter_id=adapter_id, adapter_filename=gguf_filename)


@app.delete("/v1/adapters/{adapter_id}", response_model=AdapterObject, responses={404: {"model": ErrorEnvelope}, 500: {"model": ErrorEnvelope}})
async def delete_adapter(adapter_id: str):
    if adapter_id not in _adapters:
        return _error_response(
            http_status=404,
            err_type="not_found_error",
            code="ADAPTER_NOT_FOUND",
            message="No adapter exists with that id.",
            request_id=None,
        )

    filename = _adapters.pop(adapter_id)
    return AdapterObject(adapter_id=adapter_id, adapter_filename=filename)


@app.get("/v1/adapters", response_model=AdapterListObject, responses={500: {"model": ErrorEnvelope}})
async def list_adapters():
    data = [AdapterObject(adapter_id=k, adapter_filename=v) for k, v in _adapters.items()]
    return AdapterListObject(data=data)


# Optional: simple health endpoint for local debugging (not part of the doc)
@app.get("/health")
async def health():
    try:
        r = requests.get(f"{LLAMA_BASE_URL}/health", timeout=2)
        llama_ok = r.status_code == 200
        llama_status = r.json() if llama_ok else {"error": r.text}
    except Exception as e:
        llama_ok = False
        llama_status = {"error": str(e)}

    return {
        "middleware": "ok",
        "llama_server": "ok" if llama_ok else "unreachable",
        "llama_detail": llama_status,
        "adapters_registered": len(_adapters),
    }
