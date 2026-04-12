import shutil
import sys
from pathlib import Path
import hardware
import pytest

_SETUP_DIR = str(Path(__file__).parent.parent / "setup")
if _SETUP_DIR not in sys.path:
    sys.path.insert(0, _SETUP_DIR)


_100MB = 100 * 1024 * 1024
_2GB   = 2 * 1024 ** 3
_512MB = 512 * 1024 * 1024

_PROJECT_ROOT = Path(__file__).parent.parent
_MODEL_PATH   = _PROJECT_ROOT / "llama.cpp" / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf"
_ADAPTERS_DIR = _PROJECT_ROOT / "adapters"

class TestReadBuildType:
    def test_reads_build_type(self, tmp_path):
        (tmp_path / "load.config").write_text(
            'CMAKE_ARGS="-DGGML_METAL=ON"\nBUILD_TYPE="Metal (Apple Silicon GPU acceleration)"\n'
        )
        assert hardware._read_build_type(tmp_path) == "Metal (Apple Silicon GPU acceleration)"

    def test_strips_quotes(self, tmp_path):
        (tmp_path / "load.config").write_text('BUILD_TYPE="CPU with BLAS acceleration"\n')
        result = hardware._read_build_type(tmp_path)
        assert result == "CPU with BLAS acceleration"
        assert '"' not in result

    def test_missing_file_infers_from_platform(self, tmp_path):
        result = hardware._read_build_type(tmp_path)
        assert result != ""
        assert isinstance(result, str)


class TestGetOsOverheadBytes:
    @pytest.mark.parametrize("bt", [
        "Metal (Apple Silicon GPU acceleration)",
        "CPU with Accelerate framework",
    ])
    def test_macos_returns_2gb(self, bt):
        assert hardware._get_os_overhead_bytes(bt) == _2GB

    @pytest.mark.parametrize("bt", [
        "CUDA (NVIDIA GPU acceleration)",
        "CUDA (adjust if needed)",
        "HIP/ROCm (AMD GPU acceleration)",
        "CPU with BLAS acceleration",
        "Vulkan GPU acceleration",
    ])
    def test_linux_and_gpu_returns_512mb(self, bt):
        assert hardware._get_os_overhead_bytes(bt) == _512MB

    def test_unknown_returns_2gb(self):
        assert hardware._get_os_overhead_bytes("") == _2GB
        assert hardware._get_os_overhead_bytes("CPU-only") == _2GB

class TestGetTotalRamBytes:
    def test_system_ram_returns_positive_int(self):
        build_type = hardware._read_build_type(Path(_SETUP_DIR))
        result = hardware.get_total_ram_bytes(build_type)
        assert isinstance(result, int)
        assert result > 1 * 1024 ** 3  # at least 1 GB

    @pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="nvidia-smi not available")
    def test_cuda_vram_returns_positive_int(self):
        result = hardware.get_total_ram_bytes("CUDA (NVIDIA GPU acceleration)")
        assert isinstance(result, int)
        assert result > 0

class TestGetModelSizeBytes:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x" * 1234)
        assert hardware.get_model_size_bytes(f) == 1234

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            hardware.get_model_size_bytes(tmp_path / "nonexistent.gguf")

    @pytest.mark.skipif(not _MODEL_PATH.exists(), reason="model not downloaded")
    def test_real_model_size(self):
        size = hardware.get_model_size_bytes(_MODEL_PATH)
        assert size > 1 * 1024 ** 3  # Llama 3.2 3B is well over 1 GB


class TestGetAdapterSizeEstimateBytes:
    def test_none_returns_constant(self):
        assert hardware.get_adapter_size_estimate_bytes(None) == _100MB

    def test_missing_dir_returns_constant(self, tmp_path):
        assert hardware.get_adapter_size_estimate_bytes(tmp_path / "nope") == _100MB

    def test_empty_dir_returns_constant(self, tmp_path):
        assert hardware.get_adapter_size_estimate_bytes(tmp_path) == _100MB

    def test_no_gguf_files_returns_constant(self, tmp_path):
        (tmp_path / "adapter_a").mkdir()
        (tmp_path / "adapter_a" / "config.json").write_text("{}")
        assert hardware.get_adapter_size_estimate_bytes(tmp_path) == _100MB

    def test_single_gguf_returns_its_size(self, tmp_path):
        adapter_dir = tmp_path / "adapter_a"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_a.gguf").write_bytes(b"x" * 5_000_000)
        assert hardware.get_adapter_size_estimate_bytes(tmp_path) == 5_000_000

    def test_multiple_gguf_returns_mean(self, tmp_path):
        for name, size in [("a", 10_000_000), ("b", 30_000_000)]:
            d = tmp_path / name
            d.mkdir()
            (d / f"{name}.gguf").write_bytes(b"x" * size)
        assert hardware.get_adapter_size_estimate_bytes(tmp_path) == 20_000_000

    @pytest.mark.skipif(not _ADAPTERS_DIR.exists(), reason="no adapters registered")
    def test_real_adapters_dir_returns_positive(self):
        result = hardware.get_adapter_size_estimate_bytes(_ADAPTERS_DIR)
        assert result > 0


class TestComputeMaxLoadedAdapters:
    @pytest.mark.skipif(not _MODEL_PATH.exists(), reason="model not downloaded")
    def test_real_result_within_bounds(self):
        result = hardware.compute_max_loaded_adapters(_MODEL_PATH, _ADAPTERS_DIR)
        assert hardware._MIN_CAP <= result <= hardware._MAX_CAP

    def test_missing_model_returns_fallback(self, tmp_path):
        result = hardware.compute_max_loaded_adapters(tmp_path / "missing.gguf")
        assert result == hardware._FALLBACK
