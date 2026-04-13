from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

LLAMA_BASE_URL = "http://127.0.0.1:8080"
LLAMA_SERVER_BINARY = str(_PROJECT_ROOT / "llama.cpp" / "llama-server")
LLAMA_MODEL_PATH = str(_PROJECT_ROOT / "llama.cpp" / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf")
LLAMA_ADAPTERS_DIR = str(_PROJECT_ROOT / "adapters")
LLAMA_CONVERT_SCRIPT = str(_PROJECT_ROOT / "utilities" / "convert_lora_to_gguf.py")

MAX_TOKENS_MIN = 1
MAX_TOKENS_MAX = 500
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7

# TODO: like MAX_LOADED_ADAPTERS in LlamaClient.py, this should be dynamically adjusted based on hardware
MAX_CONCURRENT_GENERATIONS = 4

EXPECTED_ARCH = "llama"

# Standard LoRA target modules for Llama, adapters involving anything outside
# this set were probably trained on a different model family
LLAMA_VALID_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "lm_head", "embed_tokens",
}