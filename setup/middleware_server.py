from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from utilities.backend_errors import (
    AdapterUploadError,
    AdapterStateError,
    BackendUnavailableError,
    BackendReloadTimeoutError,
    InferenceError,
    MalformedBackendResponseError,
)
from LlamaClient import LlamaClient  # uploaded file name
from utilities.AsyncRWLock import AsyncRWLock
from utilities.adapter_validation import validate_adapter
from utilities.constants import (
    LLAMA_BASE_URL,
    LLAMA_SERVER_BINARY,
    LLAMA_MODEL_PATH,
    LLAMA_ADAPTERS_DIR,
    LLAMA_CONVERT_SCRIPT,
MAX_TOKENS_MIN,
    MAX_TOKENS_MAX,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_CONCURRENT_GENERATIONS,
)

# creates the FastAPI application instance. This is the central object that everything else attaches to, all the
# decorators (e.g. @app.get, @app.post) throughout the file register their endpoints onto this object
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

# See above for more detail on this lock and why it is necessary
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

# Startup maintenence. We want our users to only ever have to go through the adapter registration process once. That
# means that when the server restarts it should be aware of all the adapters that were loaded in the past. So on startup
# the middleware server needs to get on the same page about that.
# Since that the middleware does not interact with the llama-server at all (and we want to keep it that way) the
# middleware is not in charge of starting up the llama-server upon its startup procedure. Just by instantiating the
# LlamaClient instance the startup process for the llama-server is triggered (occurs within the constructor for the
# object: refer to the LlamaClient comments / documentation for more information on that front).
@app.on_event("startup")
async def _autoload_adapters_from_memory():
    for filename in _llama._known_adapters:
        # Deterministic adapter_id based on filename so IDs are the same across restarts
        adapter_id = "adp_" + uuid.uuid5(uuid.NAMESPACE_URL, filename).hex[:12]
        _adapters[adapter_id] = filename

# Our application involves two servers that comprise the 'AI server' logic. The first is the middleware server (this
# file) which handles any incoming requests (as this file details), however, there is also the llama-server that the
# LlamaClient instance deals with. But it is a hassle to have to deal with both servers individually. So we abstract
# away any interaction with that back back end server. Part of that is that when the middleware server is terminated,
# it will clean up the llama-server to. We already established that the LlamaClient is in charge of starting up the
# llama-server, and this function defines the behavior that when the middleware is closed it will tell the LlamaClient
# instance to clean up that llama-server as well.
@app.on_event("shutdown")
async def _shutdown():
    _llama._server_shutdown()


# The section below is just about doing FastAPI definitions for what the API calls will be like

# START OF API SIGNATURE DEFINITION SECTION

# for defining the inner section of errors. This would contain information about the error and would contain the
# information a caller would need to properly respond to the different kinds of errors.
class ErrorInner(BaseModel):
    type: Literal["invalid_request_error", "not_found_error", "api_error"]
    code: str
    message: str
    err_id: str

# This was defined so that all errors can have a standard format that the called can expect, and then have them be
# different in the inner section
class ErrorEnvelope(BaseModel):
    error: ErrorInner

# The format of the message that the client / caller send to the middleware server when chatting
class GenerationCreateRequest(BaseModel):
    message: str = Field(..., description="The input message for the LLM to generate a response to")
    request_id: str = Field(..., description="Caller-provided identifier for tracing")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    adapter_id: Optional[str] = Field(None, description="Identifier of the adapter to apply for this generation")

# this is the format of the response that a user gets back after sending a chat query to the middleware
class GenerationObject(BaseModel):
    request_id: str
    object: Literal["generation"] = "generation"
    output: str

# This is the format of the post request that is sent to the middleware server when a caller wants to indicate that
# some files have been added to a specific directory that are (presumably) a new adapter.
class AdapterCreateRequest(BaseModel):
    adapter_dir: str = Field(..., description="Name of the adapter subdirectory inside the adapters folder")
    request_id: str = Field(..., description="Caller-provided identifier for tracing")

# This is the response that a user gets back when they (successfully) add an adapter to the known adapters of the
# local machine using a message formatted in the manner defined in AdapterCreateRequest
class AdapterObject(BaseModel):
    adapter_id: str
    adapter_filename: str
    object: Literal["adapter"] = "adapter"

# Return type for listing all the currently registered adapters
class AdapterListObject(BaseModel):
    object: Literal["list"] = "list"
    data: List[AdapterObject]

# END OF API SIGNATURE DEFINITION SECTION

# Generates an error id based on the request id. Some error handling for if none is provided, but ideally that never
# happens.
def _err_id_from_request_id(request_id: Optional[str]) -> str:
    # err_id is unique based on request id for tracing.
    if request_id:
        # Preserve the caller's id inside the err_id for easy correlation.
        return f"err_{request_id}"
    return f"err_{uuid.uuid4().hex[:10]}"

# Responsible for crafting and returning error messages to the caller who triggered them
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


# catches any HTTPException FastAPI raises outside of our logic. Normally FastAPI would return its own error format
# for these, this intercepts them and reformats them into the project's standard error envelope instead.
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

# ACTUAL REQUEST HANDLING SECTION STARTING HERE

# This function handles NON-streaming generation POST requests to the middleware server for response generation
@app.post("/v1/generations", response_model=GenerationObject, responses={
        400: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
        500: {"model": ErrorEnvelope}
    })
async def create_generation(req: GenerationCreateRequest):
    # Reject request if its for an adapter that is not on this machine
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

    # We want to use an adapter then we have to collect the associated parameters that are tracked by LlamaClient
    # as well as the additional request information (e.g. system prompt). These override the defaults associated with
    # the base model
    if req.adapter_id:
        filename = _adapters[req.adapter_id]
        system_prompt = _llama.get_system_prompt(filename)
        suggestions = _llama.get_parameter_suggestions(filename)
        if suggestions:
            temperature = suggestions.get("temperature", temperature)
            max_tokens = suggestions.get("max_tokens", max_tokens)
            min_p = suggestions.get("min_p", min_p)

    # allow for max token configuration (override of defaults)
    if req.max_tokens is not None:
        max_tokens = req.max_tokens

    # However, if the max token configuration violates the rules in place about that send an error message.
    # Note: These rules exist to place reasonable bounds on what is already a rather slow response generation
    # given the constrained hardware environment that we assume.
    if not (MAX_TOKENS_MIN <= max_tokens <= MAX_TOKENS_MAX):
        return _error_response(
            http_status=400,
            err_type="invalid_request_error",
            code="INVALID_MAX_TOKENS",
            message=f"max tokens must be between {MAX_TOKENS_MIN} and {MAX_TOKENS_MAX}.",
            request_id=req.request_id,
        )

    # Translate the adapter_id from the caller message to the actual filename associated w the adapter file
    adapter_filename = _adapters[req.adapter_id] if req.adapter_id else None

    # Get a reference to the running asyncio loop started up with the FastAPI launching. We need this for the locking
    # mechanism for parallel requests (custom semaphore defined above)
    loop = asyncio.get_running_loop()

    # wraps blocking llama.cpp call so it runs in a thread pool without blocking the event loop
    async def _run_inference():
        return await loop.run_in_executor(None, lambda: _llama.chat(
            message=req.message,
            adapter_filename=None,
            temperature=temperature,
            max_tokens=max_tokens,
            min_p=min_p,
            system_prompt=system_prompt,
        ))

    try:
        # First check avoids acquiring the write lock entirely for read-path requests (common case).
        # Second check inside the lock guards against a race where another request already switched
        # to the needed adapter between the first check and when we actually acquired the lock.
        if _needs_write(adapter_filename):
            async with _rw_lock.writing():
                global _current_inference_adapter
                if _needs_write(adapter_filename):
                    if adapter_filename:
                        await loop.run_in_executor(None, _llama.use_adapter_by_name, adapter_filename)
                    else:
                        await loop.run_in_executor(None, _llama.use_base_only)
                    _current_inference_adapter = adapter_filename
        # Write lock released before inference so concurrent readers aren't
        # blocked for the full duration of a write-path request.
        # If its just a read class request then the whole lock processing to switch adapters is skipped.
        async with _rw_lock.reading():
            output = await _run_inference()

    except BackendReloadTimeoutError as e:
        return _error_response(
            http_status=503,
            err_type="api_error",
            code="BACKEND_RELOAD_TIMEOUT",
            message=str(e),
            request_id=req.request_id,
        )
    except BackendUnavailableError as e:
        return _error_response(
            http_status=503,
            err_type="api_error",
            code="BACKEND_UNAVAILABLE",
            message=str(e),
            request_id=req.request_id,
        )
    except AdapterStateError as e:
        return _error_response(
            http_status=500,
            err_type="api_error",
            code="ADAPTER_STATE_INCONSISTENT",
            message=str(e),
            request_id=req.request_id,
        )
    except InferenceError as e:
        return _error_response(
            http_status=502,
            err_type="api_error",
            code="INFERENCE_FAILED",
            message=str(e),
            request_id=req.request_id,
        )
    except MalformedBackendResponseError as e:
        return _error_response(
            http_status=500,
            err_type="api_error",
            code="MALFORMED_BACKEND_RESPONSE",
            message=str(e),
            request_id=req.request_id,
        )
    except Exception as e:
        return _error_response(
            http_status=500,
            err_type="api_error",
            code="INTERNAL_ERROR",
            message=f"An unexpected error occurred on the backend server: {e}",
            request_id=req.request_id,
        )

    return GenerationObject(request_id=req.request_id, output=output)

# This function handles STREAMING generation POST requests to the middleware server for response generation
@app.post("/v1/generations/stream")
async def create_generation_stream(req: GenerationCreateRequest):
    # Reject request if its for an adapter that is not on this machine
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

    # We want to use an adapter then we have to collect the associated parameters that are tracked by LlamaClient
    # as well as the additional request information (e.g. system prompt). These override the defaults associated with
    # the base model
    if req.adapter_id:
        filename = _adapters[req.adapter_id]
        system_prompt = _llama.get_system_prompt(filename)
        suggestions = _llama.get_parameter_suggestions(filename)
        if suggestions:
            temperature = suggestions.get("temperature", temperature)
            max_tokens = suggestions.get("max_tokens", max_tokens)
            min_p = suggestions.get("min_p", min_p)

    # allow for max token configuration (override of defaults)
    if req.max_tokens is not None:
        max_tokens = req.max_tokens

    # However, if the max token configuration violates the rules in place about that send an error message.
    # Note: These rules exist to place reasonable bounds on what is already a rather slow response generation
    # given the constrained hardware environment that we assume.
    if not (MAX_TOKENS_MIN <= max_tokens <= MAX_TOKENS_MAX):
        return _error_response(
            http_status=400,
            err_type="invalid_request_error",
            code="INVALID_MAX_TOKENS",
            message=f"max tokens must be between {MAX_TOKENS_MIN} and {MAX_TOKENS_MAX}.",
            request_id=req.request_id,
        )

    # Translate the adapter_id from the caller message to the actual filename associated w the adapter file
    adapter_filename = _adapters[req.adapter_id] if req.adapter_id else None

    #  In the streaming version, the lock and inference logic lives inside the stream() generator, which doesn't
    #  execute when the route handler is called it executes later, when StreamingResponse starts iterating it.
    #  By that point, the route handler has already returned. So needs_write has to be evaluated before stream()
    #  is defined. If it were called inside stream() instead, it would run at stream-time rather than request-time
    needs_write = _needs_write(adapter_filename)

    # Unlike its cousin _run_inference() in the non-streaming version, the stream() function is more complex. It
    # handles its own adapter switching, locking, and inference (whereas _run_inference() lets its handler do
    # that. This is because the nature of a streaming response means that this must be a async generator.
    async def stream():
        global _current_inference_adapter

        # produces a unique object that's only purpose in life is to be unique
        # This is used to flag when the generation is done generating
        sentinel = object()

        # Get a reference to the running asyncio loop started up with the FastAPI launching. We need this for the locking
        # mechanism for parallel requests (custom semaphore defined above).
        # Unlike in the non-streaming variant this is handled inside the stream() call (as mentioned above)
        loop = asyncio.get_running_loop()

        try:
            # Lock policy is same as non-streaming version
            if needs_write:
                async with _rw_lock.writing():
                    if _needs_write(adapter_filename):
                        if adapter_filename:
                            await loop.run_in_executor(None, _llama.use_adapter_by_name, adapter_filename)
                        else:
                            await loop.run_in_executor(None, _llama.use_base_only)
                        _current_inference_adapter = adapter_filename

            # Write lock released before inference so concurrent readers aren't
            # blocked for the full duration of a write-path request.
            async with _rw_lock.reading():
                gen = await loop.run_in_executor(None, lambda: _llama.chat(
                    message=req.message,
                    adapter_filename=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    min_p=min_p,
                    system_prompt=system_prompt,
                    stream=True,
                ))

                # For reading in the data chunk by chunk as returned by the LlamaClient object
                while True:
                    chunk = await loop.run_in_executor(None, next, gen, sentinel)
                    if chunk is sentinel:
                        break
                    yield chunk

        # Headers are already sent at this point so the status code cannot change.
        # Each handler yields a JSON error token the client can detect at the end of the stream.
        except BackendReloadTimeoutError as e:
            yield json.dumps({"error": {"code": "BACKEND_RELOAD_TIMEOUT", "message": str(e)}})
        except BackendUnavailableError as e:
            yield json.dumps({"error": {"code": "BACKEND_UNAVAILABLE", "message": str(e)}})
        except AdapterStateError as e:
            yield json.dumps({"error": {"code": "ADAPTER_STATE_INCONSISTENT", "message": str(e)}})
        except InferenceError as e:
            yield json.dumps({"error": {"code": "INFERENCE_FAILED", "message": str(e)}})
        except MalformedBackendResponseError as e:
            yield json.dumps({"error": {"code": "MALFORMED_BACKEND_RESPONSE", "message": str(e)}})
        except Exception as e:
            yield json.dumps({"error": {"code": "INTERNAL_ERROR", "message": f"An unexpected error occurred: {e}"}})

    return StreamingResponse(stream(), media_type="text/plain")

# This function handles POST requests to the middleware API that want to add an adapter to the local machine's LLM
@app.post("/v1/adapters", response_model=AdapterObject, responses={400: {"model": ErrorEnvelope}, 404: {"model": ErrorEnvelope}, 500: {"model": ErrorEnvelope}})
async def create_adapter(req: AdapterCreateRequest):
    # Collect path from the request message
    adapter_path = Path(LLAMA_ADAPTERS_DIR) / req.adapter_dir

    # validate_adapter() does blocking file I/O and may invoke a subprocess to convert safetensors to .gguf,
    # so we offload it to a thread pool to avoid stalling the event loop during what could be a multi-second operation.
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, validate_adapter, adapter_path)
    except FileNotFoundError:
        # The directory indicated by the request does not exist on disk
        return _error_response(
            http_status=404,
            err_type="not_found_error",
            code="ADAPTER_DIR_NOT_FOUND",
            message=f"Directory '{req.adapter_dir}' was not found in the adapters folder.",
            request_id=req.request_id,
        )
    except AdapterUploadError as err:
        # The files are present but are invalid or incompatible with the base model.
        # AdapterUploadError is the base class for InvalidGGUFUploadError and InvalidSafetensorsUploadError,
        # so this branch catches both subtypes with the description embedded in the exception message.
        return _error_response(
            http_status=400,
            err_type="invalid_request_error",
            code="INCOMPATIBLE_ADAPTER",
            message=str(err),
            request_id=req.request_id,
        )
    except Exception as err:
        # Catch-all for anything unexpected during validation (e.g. disk errors, library failures)
        return _error_response(
            http_status=500,
            err_type="api_error",
            code="INTERNAL_ERROR",
            message=f"Adapter validation failed unexpectedly: {err}",
            request_id=req.request_id,
        )

    # Get the .gguf file (it exists for sure at this point)
    # Note: We look for this specific filename because the validate_adapter() function enforces it (contract)
    gguf_path = Path(LLAMA_ADAPTERS_DIR) / req.adapter_dir / f"{req.adapter_dir}.gguf"

    # We treat .gguf adapter files (that were originally in that form or not) the same
    gguf_filename = str(gguf_path.relative_to(Path(LLAMA_ADAPTERS_DIR)))
    adapter_id = "adp_" + uuid.uuid5(uuid.NAMESPACE_URL, gguf_filename).hex[:12] # assign a deterministic adapter id
    _adapters[adapter_id] = gguf_filename # save the mapping to the filename

    # If there is a system prompt text file take note of that
    prompt_file = adapter_path / "system_prompt.txt"
    system_prompt = prompt_file.read_text().strip() if prompt_file.exists() else None

    # also take note if a parameter suggestions file was uploaded
    suggestions_file = adapter_path / "parameter_suggestions.json"
    parameter_suggestions = None
    if suggestions_file.exists():
        try:
            raw = json.loads(suggestions_file.read_text())
            parameter_suggestions = {k: v for k, v in raw.items() if k in {"temperature", "min_p", "max_tokens"}}
        except ValueError:
            pass  # malformed json, silently ignore, adapter still registers without suggestions

    # and now with all the collected information we can actually register the adapter with the LlamaClient instance
    _llama.register_adapter(gguf_filename, system_prompt=system_prompt, parameter_suggestions=parameter_suggestions)

    # and return a successful confirmation message to the caller
    return AdapterObject(adapter_id=adapter_id, adapter_filename=gguf_filename)


# We also needed to modify the deletion logic to accomodate parallel execution
# If you want to delete an adapter that is not the currently active one, that is fine and can
# be done in parallel with a generation request. What you cannot do is delete an adapter out from under
# a process.
@app.delete("/v1/adapters/{adapter_id}", response_model=AdapterObject, responses={404: {"model": ErrorEnvelope}, 500: {"model": ErrorEnvelope}})
async def delete_adapter(adapter_id: str):
    # If you try to delete a non-existent adapter you get an error
    if adapter_id not in _adapters:
        return _error_response(
            http_status=404,
            err_type="not_found_error",
            code="ADAPTER_NOT_FOUND",
            message="No adapter exists with that id.",
            request_id=None,
        )

    # get the filename from the mapping of id to filename
    filename = _adapters[adapter_id]

    # If the adapter is currently mounted onto the LLM instance then you need to acquire the write lock,
    # as this process could interfere with currently operating generations
    if filename in _llama._active_adapters:
        async with _rw_lock.writing():
            _adapters.pop(adapter_id)
            _llama.unregister_adapter(filename) # LlamaClient handles the actual deregistration logic
            global _current_inference_adapter
            if _current_inference_adapter == filename:
                _current_inference_adapter = None
    # But if the adapter was not currently mounted to the LLM then no problem. Note: this may cause issues if there
    # were queued requests that wanted to use that adapter, but that scenario is pretty unlikely, and would return an
    # error that the caller should be equipped to handle.
    else:
        _adapters.pop(adapter_id)
        _llama.unregister_adapter(filename)

    return AdapterObject(adapter_id=adapter_id, adapter_filename=filename)

# This returns all adapters that are known to have been registered with the middleware
# TODO: Could probably do this through the LlamaClient, let there be a single point of knowledge there
@app.get("/v1/adapters", response_model=AdapterListObject, responses={500: {"model": ErrorEnvelope}})
async def list_adapters():
    data = [AdapterObject(adapter_id=k, adapter_filename=v) for k, v in _adapters.items()]
    return AdapterListObject(data=data)

# Need this to know is all is well. Used on application startup to poll to see when the server is ready to accept
# queries. Specifically checking if the back back end is up and running, as that takes longer than the middleware
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

# ACTUAL REQUEST HANDLING SECTION ENDING HERE

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=4000, log_level='trace')
