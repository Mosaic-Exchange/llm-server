# Setup & Usage Guide

## Prerequisites

- Python 3.8+
- CMake
- Git
- `curl`

---

## 1. Build llama.cpp & Download the Model

Run the setup script from the repo root. It will install Python dependencies, detect your hardware, build llama.cpp with the appropriate acceleration (CUDA, ROCm, Metal, etc.), and download the Llama 3.2 3B model automatically.

```bash
bash setup/setup_llama_cpp.sh
```

> **Note:** If running on a new machine, make sure `setup/load.config` does not exist beforehand — it caches hardware settings from a previous run and will skip hardware detection if present.
>
> ```bash
> rm -f setup/load.config
> ```

This will produce:
- `llama.cpp/llama-server` — the inference server binary
- `llama.cpp/convert_lora_to_gguf.py` — adapter conversion script
- `llama.cpp/models/llama-3.2-3b-instruct-q4_k_m.gguf` — the model
- `adapters/` — created at the repo root if not already present

---

## 2. Start the Middleware Server

The middleware server manages the llama.cpp process, handles LoRA adapter registration, and exposes the API.

```bash
cd setup
python -m uvicorn middleware_server:app --host 127.0.0.1 --port 4000
```

On startup it will automatically launch `llama-server` on port 8080 if it is not already running.

---

## 3. API Endpoints

Base URL: `http://127.0.0.1:4000`

### Generate a response

```
POST /v1/generations
```

```json
{
  "message": "Your prompt here",
  "request_id": "req_001",
  "max_tokens": 256,
  "adapter_id": "adp_..."
}
```

`adapter_id` is optional. If omitted, the base model is used.

---

### Register an adapter

```
POST /v1/adapters
```

```json
{
  "adapter_dir": "my_adapter",
  "request_id": "req_002"
}
```

`adapter_dir` must be a subdirectory inside `adapters/` containing a HuggingFace PEFT adapter (`adapter_config.json` + `adapter_model.safetensors`). The server will convert it to GGUF automatically.

---

### List registered adapters

```
GET /v1/adapters
```

---

### Remove an adapter

```
DELETE /v1/adapters/{adapter_id}
```

---

### Health check

```
GET /health
```

Returns the status of both the middleware and the underlying llama.cpp server.

---

## 4. Placing Adapters

Put HuggingFace PEFT adapter directories inside `adapters/` at the repo root:

```
adapters/
  my_adapter/
    adapter_config.json
    adapter_model.safetensors
```

Then register via `POST /v1/adapters` with `"adapter_dir": "my_adapter"`.

Optionally include in the adapter directory:
- `system_prompt.txt` — a system prompt applied automatically when this adapter is used
- `parameter_suggestions.json` — default generation parameters, e.g.:
  ```json
  { "temperature": 0.5, "min_p": 0.05, "max_tokens": 300 }
  ```

---

## 5. Troubleshooting

### Port already in use

If the middleware fails to start with a timeout error, a leftover `llama-server` process may still be holding port 8080. Check and kill it:

```bash
lsof -i :8080 -i :4000
kill <PID>
```

---

## 6. Testing

Test the middleware server:

```bash
python testing/test_client.py
```

Interactive chat session:

```bash
python testing/interactive_chat.py
```
