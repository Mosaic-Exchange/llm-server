import logging
import platform
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("uvicorn")

_ADAPTER_SIZE_ESTIMATE_BYTES = 100 * 1024 * 1024   
_INFERENCE_OVERHEAD_BYTES    = int(1.5 * 1024 ** 3)
_BUDGET_FRACTION             = 0.50
_MAX_CAP                     = 10
_MIN_CAP                     = 1
_FALLBACK                    = 2


def _infer_build_type_from_platform() -> str:
    # no load config 
    if sys.platform == "darwin":
        if platform.machine() == "arm64":
            return "Metal (Apple Silicon GPU acceleration)"
        return "CPU with Accelerate framework"
    if sys.platform.startswith("linux"):
        return "CPU with BLAS acceleration"
    if sys.platform == "win32":
        return "CUDA (adjust if needed)"
    return "CPU-only"


def _read_build_type(setup_dir: Path) -> str:
    # read from load config
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
        return 512 * 1024 * 1024 # headless linux
    return 2 * 1024 ** 3 # in case it is unknown


def get_total_ram_bytes(build_type: str) -> int:
   # get total ram for computation based on the type of hardware from load.config
    bt = build_type.lower()

    if "metal" in bt or "accelerate" in bt:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip())

    if "cuda" in bt:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        mib = int(result.stdout.strip().splitlines()[0])
        return mib * 1024 * 1024

    if "hip" in bt or "rocm" in bt or "blas" in bt or "vulkan" in bt or "cpu" in bt:
        return _read_proc_meminfo()

    # load.config absent
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip())

    if sys.platform.startswith("linux"):
        return _read_proc_meminfo()

    if sys.platform == "win32":
        return _read_windows_ram()

    raise RuntimeError(f"unsupported platform: {sys.platform!r}")


def _read_proc_meminfo() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb * 1024
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def _read_windows_ram() -> int:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength",                ctypes.c_ulong),
            ("dwMemoryLoad",            ctypes.c_ulong),
            ("ullTotalPhys",            ctypes.c_ulonglong),
            ("ullAvailPhys",            ctypes.c_ulonglong),
            ("ullTotalPageFile",        ctypes.c_ulonglong),
            ("ullAvailPageFile",        ctypes.c_ulonglong),
            ("ullTotalVirtual",         ctypes.c_ulonglong),
            ("ullAvailVirtual",         ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    ms = MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(ms)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullTotalPhys


def get_model_size_bytes(model_path) -> int:
    return Path(model_path).stat().st_size


def get_adapter_size_estimate_bytes(adapters_dir) -> int:
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
    # apply formula
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