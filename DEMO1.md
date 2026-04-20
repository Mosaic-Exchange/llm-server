## 1. Build llama.cpp & Download the Model

Run the setup script from the repo root. It will install Python dependencies, detect your hardware, build llama.cpp with the appropriate acceleration (CUDA, ROCm, Metal, etc.), and download the Llama 3.2 3B model automatically.

```bash
bash setup/setup_llama_cpp.sh
```

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

## 3. Copy Over Some Adapters for Demonstration

```bash
for d in /Users/danielarturi/Desktop/COMP_361_Personal/llama.cpp/adapter_backup/*/; do cp -r "${d%/}" /Users/danielarturi/Desktop/361_Group/llm-server/adapters/; done
```

In the real execution of the program this import would be handled by a different part of the program, I am just doing this to demonstrate

---

## 4. Register Adapters

```bash
curl -X POST http://127.0.0.1:4000/v1/adapters \
  -H "Content-Type: application/json" \
  -d '{"adapter_dir": "ella", "request_id": "reg-ella"}'  
```

```bash
curl -X POST http://127.0.0.1:4000/v1/adapters \
  -H "Content-Type: application/json" \
  -d '{"adapter_dir": "gordon_ramsey", "request_id": "reg-gordon"}'
```

```bash
curl -X POST http://127.0.0.1:4000/v1/adapters \
  -H "Content-Type: application/json" \
  -d '{"adapter_dir": "structured_answer_adapter", "request_id": "reg-structured"}'
```

#### 4.2 - Messed up adapter upload
```bash
curl -X POST http://127.0.0.1:4000/v1/adapters \
  -H "Content-Type: application/json" \
  -d '{"adapter_dir": "pioneer", "request_id": "reg-pioneer"}'  
```

## 5. Query Gordon Ramsey

```bash
curl -X POST http://127.0.0.1:4000/v1/generations \
  -H "Content-Type: application/json" \
  -d '{"message": "How are you?", "request_id": "test-1", "adapter_id": "adp_ad28b0db0dfe"}'
```

## 6. Parallel Gordon Stream Query

```bash
curl -N -X POST http://127.0.0.1:4000/v1/generations/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "write me a long poem", "request_id": "test-1", "adapter_id": "adp_ad28b0db0dfe"}'
```

```bash
curl -N -X POST http://127.0.0.1:4000/v1/generations/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your favorite turkey dish?", "request_id": "test-1", "adapter_id": "adp_ad28b0db0dfe"}'
```

## 6. Parallel Gordon / Ella Stream Query

```bash
curl -N -X POST http://127.0.0.1:4000/v1/generations/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "write me a long poem", "request_id": "test-1", "adapter_id": "adp_ad28b0db0dfe"}'
```

```bash
curl -N -X POST http://127.0.0.1:4000/v1/generations/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your favorite turkey dish?", "request_id": "test-1", "adapter_id": "adp_2a2acd1ecbce"}'
```

## 7. View Adapters
```bash
curl http://127.0.0.1:4000/v1/adapters
```

## RESET DEMO1
```bash
bash /Users/danielarturi/Desktop/361_Group/llm-server/utilities/reset_demo.sh
```