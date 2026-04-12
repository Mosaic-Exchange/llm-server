import logging
import sys
import platform
import subprocess
from pathlib import Path
import psutil

logger = logging.getLogger("uvicorn")

_ADAPTER_SIZE_ESTIMATE_BYTES = 100 * 1024 * 1024
_INFERENCE_OVERHEAD_BYTES    = int(1.5 * 1024 ** 3)
_BUDGET_FRACTION             = 0.50
_MAX_CAP                     = 10
_MIN_CAP                     = 1
_FALLBACK                    = 2


def _infer_build_type_from_platform() -> str:
    # no load config
    current_platform = sys.platform
    if current_platform == "darwin":
        if platform.machine() == "arm64":
            return "Metal (Apple Silicon GPU acceleration)"
        return "CPU with Accelerate framework"
    elif current_platform.startswith("linux"):
        return "CPU with BLAS acceleration"
    elif current_platform == "win32":
        return "CUDA (adjust if needed)"
    else:
        return "CPU-only"


def _read_build_type(setup_dir: Path) -> str:
    #read from load config
    config_path = setup_dir / "load.config"
    try:
        for line in config_path.read_text().splitlines():
            if line.startswith("BUILD_TYPE="):
                value = line[len("BUILD_TYPE="):].strip()
                return value.strip('"')
    except Exception:
        pass
    return _infer_build_type_from_platform()


def _get_os_overhead_bytes(build_type: str) -> int:
    # get OS overhead
    bt = build_type.lower()
    if "metal" in bt or "accelerate" in bt:
        return 2 * 1024 ** 3 #macOS
    if "cuda" in bt or "hip" in bt or "rocm" in bt or "vulkan" in bt or "blas" in bt:
        return 512 * 1024 * 1024 #headless linux
    return 2 * 1024 ** 3 # in case it is unknown


def get_total_ram_bytes(build_type: str) -> int:
    #get total ram for computation based on the type of hardware from load.config
    if "cuda" in build_type.lower():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        mib = int(result.stdout.strip().splitlines()[0])
        return mib * 1024 * 1024

    return psutil.virtual_memory().total


def get_model_size_bytes(model_path) -> int:
    return Path(model_path).stat().st_size


def get_adapter_size_estimate_bytes(adapters_dir) -> int:
    #make average of adapters if they exist
    if adapters_dir is None:
        return _ADAPTER_SIZE_ESTIMATE_BYTES
    p = Path(adapters_dir)
    if not p.exists() or not p.is_dir():
        return _ADAPTER_SIZE_ESTIMATE_BYTES
    gguf_files = list(p.rglob("*.gguf"))
    if not gguf_files:
        return _ADAPTER_SIZE_ESTIMATE_BYTES
    sizes = [f.stat().st_size for f in gguf_files]
    return max(1, int(sum(sizes) / len(sizes)))


def compute_max_loaded_adapters(model_path, adapters_dir=None, setup_dir=None) -> int:
    #apply formula and compute
    try:
        _setup_dir   = Path(setup_dir) if setup_dir else Path(__file__).parent
        build_type   = _read_build_type(_setup_dir)
        total_ram    = get_total_ram_bytes(build_type)
        os_overhead  = _get_os_overhead_bytes(build_type)
        model_size   = get_model_size_bytes(model_path)
        per_adapter  = get_adapter_size_estimate_bytes(adapters_dir)

        remaining = total_ram - model_size - os_overhead - _INFERENCE_OVERHEAD_BYTES
        if remaining <= 0:
            return _MIN_CAP

        raw    = int(remaining * _BUDGET_FRACTION // per_adapter)
        result = max(_MIN_CAP, min(raw, _MAX_CAP))
        logger.info(
            "MAX_LOADED_ADAPTERS=%d  (build=%r  total=%.1fGB  model=%.1fGB"
            "  os_overhead=%.1fGB  per_adapter=%.0fMB)",
            result, build_type,
            total_ram / 1e9, model_size / 1e9,
            os_overhead / 1e9, per_adapter / 1e6,
        )
        return result
    except Exception:
        logger.warning(
            "hardware.py: could not compute adapter limit, falling back to %d",
            _FALLBACK, exc_info=True,
        )
        return _FALLBACK