import sys
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

_SETUP_DIR = str(Path(__file__).parent.parent / "setup")
if _SETUP_DIR not in sys.path:
    sys.path.insert(0, _SETUP_DIR)

import hardware

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


_2GB  = 2 * 1024 ** 3
_512MB = 512 * 1024 * 1024


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
    def test_macos_metal_sysctl(self):
        with patch("hardware.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "17179869184\n"
            result = hardware.get_total_ram_bytes("Metal (Apple Silicon GPU acceleration)")
        assert result == 17179869184

    def test_macos_accelerate_sysctl(self):
        with patch("hardware.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "8589934592\n"
            result = hardware.get_total_ram_bytes("CPU with Accelerate framework")
        assert result == 8589934592

    def test_cuda_nvidia_smi(self):
        with patch("hardware.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "8192\n"
            result = hardware.get_total_ram_bytes("CUDA (NVIDIA GPU acceleration)")
        assert result == 8192 * 1024 * 1024

    def test_cuda_windows(self):
        with patch("hardware.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "4096\n"
            result = hardware.get_total_ram_bytes("CUDA (adjust if needed)")
        assert result == 4096 * 1024 * 1024

    def test_linux_blas_proc_meminfo(self):
        meminfo = "MemTotal:       16777216 kB\nMemFree:         1234567 kB\n"
        with patch("builtins.open", mock_open(read_data=meminfo)):
            result = hardware.get_total_ram_bytes("CPU with BLAS acceleration")
        assert result == 16777216 * 1024

    def test_linux_hip_proc_meminfo(self):
        meminfo = "MemTotal:        8388608 kB\n"
        with patch("builtins.open", mock_open(read_data=meminfo)):
            result = hardware.get_total_ram_bytes("HIP/ROCm (AMD GPU acceleration)")
        assert result == 8388608 * 1024

    def test_fallback_sys_platform_darwin(self):
        with patch("hardware.sys") as mock_sys, patch("hardware.subprocess.run") as mock_run:
            mock_sys.platform = "darwin"
            mock_run.return_value.stdout = "17179869184\n"
            result = hardware.get_total_ram_bytes("")
        assert result == 17179869184

    def test_fallback_sys_platform_linux(self):
        meminfo = "MemTotal:       16777216 kB\n"
        with patch("hardware.sys") as mock_sys, patch("builtins.open", mock_open(read_data=meminfo)):
            mock_sys.platform = "linux"
            result = hardware.get_total_ram_bytes("")
        assert result == 16777216 * 1024

    def test_fallback_unknown_platform_raises(self):
        with patch("hardware.sys") as mock_sys:
            mock_sys.platform = "freebsd"
            with pytest.raises(RuntimeError, match="unsupported platform"):
                hardware.get_total_ram_bytes("")

class TestGetModelSizeBytes:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x" * 1234)
        assert hardware.get_model_size_bytes(f) == 1234

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            hardware.get_model_size_bytes(tmp_path / "nonexistent.gguf")


_100MB = 100 * 1024 * 1024


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
        result = hardware.get_adapter_size_estimate_bytes(tmp_path)
        assert result == 20_000_000


_MODEL  = 1_900 * 1024 * 1024   # 1.9 GB
_INFR   = hardware._INFERENCE_OVERHEAD_BYTES


class TestComputeMaxLoadedAdapters:
    def _patch_all(self, total_ram, build_type="CPU with BLAS acceleration",
                   model_size=_MODEL, per_adapter=_100MB):
        """Context manager that patches all sub-functions."""
        return (
            patch("hardware._read_build_type", return_value=build_type),
            patch("hardware.get_total_ram_bytes", return_value=total_ram),
            patch("hardware.get_model_size_bytes", return_value=model_size),
            patch("hardware.get_adapter_size_estimate_bytes", return_value=per_adapter),
        )

    def _run(self, total_ram, build_type="CPU with BLAS acceleration",
             model_size=_MODEL, per_adapter=_100MB):
        with patch("hardware._read_build_type", return_value=build_type), \
             patch("hardware.get_total_ram_bytes", return_value=total_ram), \
             patch("hardware.get_model_size_bytes", return_value=model_size), \
             patch("hardware.get_adapter_size_estimate_bytes", return_value=per_adapter):
            return hardware.compute_max_loaded_adapters("/fake/model.gguf")

    def test_typical_macos_16gb(self):
        result = self._run(16 * 1024 ** 3, build_type="Metal (Apple Silicon GPU acceleration)")
        assert result == hardware._MAX_CAP

    def test_linux_6gb_hits_cap(self):
        # Linux overhead is 512 MB, so remaining is generous even on 6 GB
        result = self._run(6 * 1024 ** 3, build_type="CPU with BLAS acceleration")
        assert result == hardware._MAX_CAP

    def test_low_ram_macos_6gb(self):
        # macOS overhead is 2 GB; remaining = 6G - 1.9G - 2G - 1.5G = 0.6 GB
        # budget = 0.3 GB → floor(0.3G / 100M) = 3
        result = self._run(6 * 1024 ** 3, build_type="Metal (Apple Silicon GPU acceleration)")
        assert result == 3

    def test_very_low_ram_returns_minimum(self):
        # 4 GB macOS: remaining goes negative
        result = self._run(4 * 1024 ** 3, build_type="Metal (Apple Silicon GPU acceleration)")
        assert result == hardware._MIN_CAP

    def test_result_never_exceeds_cap(self):
        result = self._run(256 * 1024 ** 3)
        assert result == hardware._MAX_CAP

    def test_result_never_below_minimum(self):
        result = self._run(1 * 1024 ** 3, build_type="Metal (Apple Silicon GPU acceleration)")
        assert result == hardware._MIN_CAP

    def test_model_missing_returns_fallback(self):
        with patch("hardware._read_build_type", return_value="CPU with BLAS acceleration"), \
             patch("hardware.get_total_ram_bytes", return_value=16 * 1024 ** 3), \
             patch("hardware.get_model_size_bytes", side_effect=FileNotFoundError):
            result = hardware.compute_max_loaded_adapters("/missing/model.gguf")
        assert result == hardware._FALLBACK

    def test_ram_error_returns_fallback(self):
        with patch("hardware._read_build_type", return_value="CPU with BLAS acceleration"), \
             patch("hardware.get_total_ram_bytes", side_effect=RuntimeError("no ram")):
            result = hardware.compute_max_loaded_adapters("/fake/model.gguf")
        assert result == hardware._FALLBACK

    def test_load_config_absent_then_ram_error_returns_fallback(self):
        # load.config absent so empty build_type. get_total_ram_bytes will raise on unknown platform
        with patch("hardware._read_build_type", return_value=""), \
             patch("hardware.get_total_ram_bytes", side_effect=RuntimeError("unsupported")):
            result = hardware.compute_max_loaded_adapters("/fake/model.gguf")
        assert result == hardware._FALLBACK
