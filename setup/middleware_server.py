from __future__ import annotations

import asyncio
import contextlib # This library is for managing the parallel generation requests
import json
import logging
import uuid
from typing import Any, Dict, List, Literal, Optional

import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from LlamaClient import LlamaClient  # uploaded file name


LLAMA_BASE_URL = "http://127.0.0.1:8080"
_PROJECT_ROOT = Path(__file__).parent.parent
LLAMA_SERVER_BINARY = str(_PROJECT_ROOT / "llama.cpp" / "llama-server")
LLAMA_MODEL_PATH = str(_PROJECT_ROOT / "llama.cpp" / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf")
LLAMA_ADAPTERS_DIR = str(_PROJECT_ROOT / "adapters")
LLAMA_CONVERT_SCRIPT = str(_PROJECT_ROOT / "utilities" / "convert_lora_to_gguf.py")
MAX_TOKENS_MIN = 1
MAX_TOKENS_MAX = 500
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
MAX_CONCURRENT_GENERATIONS = 4 # TODO: Like the 'MAX_LOADED_ADAPTERS' field in the LlamaClient.py file we want to make this dynamically adjusted to the machine that the program is being run on

# This is a custom lock we defined for the following purpose:
# We want to enable parallel generation requests, however there are limitations on that
# Certain generation requests are going to trigger the model reloading to load up that adapter
# So while we can allow parallel generation requests if the requests after the first active one target
# adapters that are already loaded, we cannot allow concurrency for requests that would force
# a reload in the middle of a generation process (sweep the current model out from under a generation request).
# Thus we define a lock to help implement this lock. Additionally we define a maximum number of requests that
# can be active at a given time as to not overwhelm the hardware. This is implemented in a semaphore.
class AsyncRWLock:
    def __init__(self, max_readers: int):
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer_active = False
        self._writer_waiting = 0
        self._max_readers = max_readers

    @contextlib.asynccontextmanager
    async def reading(self):
        async with self._cond:
            # Check all the conditions for a read process
            # We give writers priority over readers to prevent starvation
            # NOTE: This means that a queued read could come to find that when it gets where it wanted to go
            # the adapter it expected is gone
            while self._writer_active or self._writer_waiting > 0 or self._readers >= self._max_readers:
                await self._cond.wait()
            self._readers += 1 # Increment to count against the MAX_CONCURRENT_GENERATIONS field / value
        try:
            yield
        finally:
            async with self._cond:
                self._readers -= 1 # Decrement to count against the MAX_CONCURRENT_GENERATIONS field / value
                self._cond.notify_all()

    @contextlib.asynccontextmanager
    async def writing(self):
        async with self._cond:
            self._writer_waiting += 1 # Increment to count against the MAX_CONCURRENT_GENERATIONS field / value
            try:

                while self._writer_active or self._readers > 0:
                    await self._cond.wait()
                self._writer_active = True
            finally:
                self._writer_waiting -= 1 # Decrement to count against the MAX_CONCURRENT_GENERATIONS field / value
        try:
            yield
        finally:
            async with self._cond:
                self._writer_active = False
                self._cond.notify_all()


app = FastAPI(title="mosaicAI Middleware", version="1.0")

# This is essentially the object that the middleware is in charge of managing
# and interacting with
_llama = LlamaClient(
    base_url=LLAMA_BASE_URL,
    server_binary=LLAMA_SERVER_BINARY,
    model_path=LLAMA_MODEL_PATH,
    adapters_dir=LLAMA_ADAPTERS_DIR,
    convert_script=LLAMA_CONVERT_SCRIPT,
)

# Adapter registry (in-memory): adapter_id -> adapter_filename
# When the server is reloaded this can be populated using the information
# That was written in the json file
_adapters: Dict[str, str] = {}

_rw_lock = AsyncRWLock(max_readers=MAX_CONCURRENT_GENERATIONS)
_current_inference_adapter: Optional[str] = None  # adapter filename at scale 1.0, or None = base model

# A 'read' request is when no state change is needed to the model
# this is when the adapter requested is the one that is actively loaded (so no model reload is needed)
# and currently active (so the scale of the model adapters reflects that)
# We let such requests be parallel, but not otherwise.
def _needs_write(adapter_filename: Optional[str]) -> bool:
    if adapter_filename == _current_inference_adapter:
        if adapter_filename is None or adapter_filename in _llama._active_adapters:
            return False
    return True


@app.on_event("startup")
async def _autoload_adapters_from_memory():
    for filename in _llama._known_adapters:
        # Deterministic adapter_id based on filename so IDs are the same across restarts
        adapter_id = "adp_" + uuid.uuid5(uuid.NAMESPACE_URL, filename).hex[:12]
        _adapters[adapter_id] = filename


@app.on_event("shutdown")
async def _shutdown():
    import subprocess
    if _llama._server_process and _llama._server_process.poll() is None:
        _llama._server_process.terminate()
        try:
            _llama._server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _llama._server_process.kill()


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


_ADAPTER_WEIGHTS_OPTIONS = {"adapter_model.safetensors", "adapter_model.bin"}

def _validate_adapter_dir(adapter_path: Path) -> Optional[str]:
    missing = []
    if not (adapter_path / "adapter_config.json").exists():
        missing.append("adapter_config.json")
    if not any((adapter_path / f).exists() for f in _ADAPTER_WEIGHTS_OPTIONS):
        missing.append("adapter_model.safetensors (or adapter_model.bin)")
    return f"Missing required files: {', '.join(missing)}" if missing else None


@app.post("/v1/generations", response_model=GenerationObject, responses={400: {"model": ErrorEnvelope}, 404: {"model": ErrorEnvelope}, 500: {"model": ErrorEnvelope}})
async def create_generation(req: GenerationCreateRequest):
    if req.adapter_id and req.adapter_id not in _adapters:
        return _error_response(
            http_status=404,
            err_type="not_found_error",
            code="ADAPTER_NOT_FOUND",
            message="The specified adapter_id does not exist.",
            request_id=req.request_id,
        )

    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS
    min_p = None
    system_prompt = None

    if req.adapter_id:
        filename = _adapters[req.adapter_id]
        system_prompt = _llama.get_system_prompt(filename)
        suggestions = _llama.get_parameter_suggestions(filename)
        if suggestions:
            temperature = suggestions.get("temperature", temperature)
            max_tokens = suggestions.get("max_tokens", max_tokens)
            min_p = suggestions.get("min_p", min_p)

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

    adapter_filename = _adapters[req.adapter_id] if req.adapter_id else None
    loop = asyncio.get_running_loop()

    async def _run_inference():
        return await loop.run_in_executor(None, lambda: _llama.chat(
            message=req.message,
            adapter_filename=None,
            temperature=temperature,
            max_tokens=max_tokens,
            min_p=min_p,
            system_prompt=system_prompt,
        ))

    # Use run_in_executor to create thread pool execution so event loop doesnt need to block
    try:
        if _needs_write(adapter_filename):
            async with _rw_lock.writing():
                global _current_inference_adapter
                if _needs_write(adapter_filename):
                    # We do a zero argument lambda to avoid unpacking args
                    # this if else is j about setting the adapter
                    if adapter_filename:
                        await loop.run_in_executor(None, _llama.use_adapter_by_name, adapter_filename)
                    else:
                        await loop.run_in_executor(None, _llama.use_base_only)
                    _current_inference_adapter = adapter_filename
                # here is where we generate an answer (for write processes)
                output = await _run_inference()
        else:
            async with _rw_lock.reading():
                # here is where we generate an answer (for read processes)
                output = await _run_inference()
    except Exception as e:
        return _error_response(
            http_status=500,
            err_type="api_error",
            code="INTERNAL_ERROR",
            message=f"An unexpected error occurred on the backend server: {e}",
            request_id=req.request_id,
        )

    return GenerationObject(request_id=req.request_id, output=output)


@app.post("/v1/generations/stream")
async def create_generation_stream(req: GenerationCreateRequest):
    if req.adapter_id and req.adapter_id not in _adapters:
        return _error_response(
            http_status=404,
            err_type="not_found_error",
            code="ADAPTER_NOT_FOUND",
            message="The specified adapter_id does not exist.",
            request_id=req.request_id,
        )

    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS
    min_p = None
    system_prompt = None

    if req.adapter_id:
        filename = _adapters[req.adapter_id]
        system_prompt = _llama.get_system_prompt(filename)
        suggestions = _llama.get_parameter_suggestions(filename)
        if suggestions:
            temperature = suggestions.get("temperature", temperature)
            max_tokens = suggestions.get("max_tokens", max_tokens)
            min_p = suggestions.get("min_p", min_p)

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

    adapter_filename = _adapters[req.adapter_id] if req.adapter_id else None
    needs_write = _needs_write(adapter_filename)

    async def stream():
        global _current_inference_adapter
        sentinel = object()
        loop = asyncio.get_running_loop()

        # Make sure to acquire appropriate lock
        # See the non-stream version for a more verbose comment job (similar pattern)
        async with (_rw_lock.writing() if needs_write else _rw_lock.reading()):
            if needs_write and _needs_write(adapter_filename):
                if adapter_filename:
                    await loop.run_in_executor(None, _llama.use_adapter_by_name, adapter_filename)
                else:
                    await loop.run_in_executor(None, _llama.use_base_only)
                _current_inference_adapter = adapter_filename
            gen = await loop.run_in_executor(None, lambda: _llama.chat(
                message=req.message,
                adapter_filename=None,
                temperature=temperature,
                max_tokens=max_tokens,
                min_p=min_p,
                system_prompt=system_prompt,
                stream=True,
            ))
            while True:
                chunk = await loop.run_in_executor(None, next, gen, sentinel)
                if chunk is sentinel:
                    break
                yield chunk

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/v1/adapters", response_model=AdapterObject, responses={400: {"model": ErrorEnvelope}, 404: {"model": ErrorEnvelope}, 500: {"model": ErrorEnvelope}})
async def create_adapter(req: AdapterCreateRequest):
    adapter_path = Path(LLAMA_ADAPTERS_DIR) / req.adapter_dir

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
                    _cfg = json.load(f)
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

        msg = _validate_adapter_dir(adapter_path)
        if msg:
            return _error_response(
                http_status=400,
                err_type="invalid_request_error",
                code="MISSING_ADAPTER_FILES",
                message=msg,
                request_id=req.request_id,
            )

        loop = asyncio.get_running_loop()
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

        # Delete weight files as they are no longer needed once GGUF exists
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
            raw = json.loads(suggestions_file.read_text())
            parameter_suggestions = {k: v for k, v in raw.items() if k in {"temperature", "min_p", "max_tokens"}}
        except ValueError:
            pass  # malformed json, silently ignore, adapter still registers without suggestions

    _llama.register_adapter(gguf_filename, system_prompt=system_prompt, parameter_suggestions=parameter_suggestions)

    return AdapterObject(adapter_id=adapter_id, adapter_filename=gguf_filename)


# We also need to modify the deletion logic to accomodate parallel execution
# If you want to delete an adapter that is not the currently active one, that is fine and can
# be done in parallel with a generation request. What you cannot do is delete an adapter out from under
# a process.
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

    filename = _adapters[adapter_id]
    if filename in _llama._active_adapters:
        async with _rw_lock.writing():
            _adapters.pop(adapter_id)
            _llama.unregister_adapter(filename)
            global _current_inference_adapter
            if _current_inference_adapter == filename:
                _current_inference_adapter = None
    else:
        _adapters.pop(adapter_id)
        _llama.unregister_adapter(filename)
    return AdapterObject(adapter_id=adapter_id, adapter_filename=filename)


@app.get("/v1/adapters", response_model=AdapterListObject, responses={500: {"model": ErrorEnvelope}})
async def list_adapters():
    data = [AdapterObject(adapter_id=k, adapter_filename=v) for k, v in _adapters.items()]
    return AdapterListObject(data=data)


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


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=4000, log_level='trace')
