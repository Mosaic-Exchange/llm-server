#!/bin/bash
# setup_llama_cpp.sh - Automatic llama.cpp setup with hardware detection (CMake version)

set -e  # Exit on error

echo "=== llama.cpp Automatic Setup Script (CMake) ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for CMake
check_cmake() {
    if ! command -v cmake &> /dev/null; then
        echo -e "${RED}✗ CMake not found${NC}"
        echo "Please install CMake:"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  brew install cmake"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "  sudo apt-get install cmake  # Debian/Ubuntu"
            echo "  sudo dnf install cmake      # Fedora"
        fi
        exit 1
    else
        CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
        echo -e "${GREEN}✓ CMake found (version $CMAKE_VERSION)${NC}"
    fi
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Detect NVIDIA GPU
detect_nvidia() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            return 0
        fi
    fi
    return 1
}

# Detect AMD GPU
detect_amd() {
    if command -v rocm-smi &> /dev/null; then
        echo -e "${GREEN}✓ AMD GPU detected (ROCm available)${NC}"
        return 0
    elif lspci 2>/dev/null | grep -i "VGA.*AMD" &> /dev/null; then
        echo -e "${YELLOW}⚠ AMD GPU detected but ROCm not found${NC}"
        return 1
    fi
    return 1
}

# Detect Apple Silicon
detect_apple_silicon() {
    if [[ $(uname -s) == "Darwin" ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            echo -e "${GREEN}✓ Apple Silicon detected${NC}"
            sysctl -n machdep.cpu.brand_string
            return 0
        fi
    fi
    return 1
}

# Detect Vulkan support
detect_vulkan() {
    if command -v vulkaninfo &> /dev/null; then
        if vulkaninfo &> /dev/null; then
            echo -e "${GREEN}✓ Vulkan support detected${NC}"
            return 0
        fi
    fi
    return 1
}

# Config file path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/load.config"
PROJECT_DIR="$SCRIPT_DIR/.."

# Check for cached configuration
if [ -f "$CONFIG_FILE" ]; then
    echo "Found cached configuration at $CONFIG_FILE"
    source "$CONFIG_FILE"
    echo -e "${GREEN}✓ Loaded CMAKE_ARGS and BUILD_TYPE from cache${NC}"
else
    # Main detection logic
    echo "Detecting hardware configuration..."
    echo ""

    check_cmake

    OS=$(detect_os)
    CMAKE_ARGS=""
    BUILD_TYPE="CPU-only"

    case $OS in
        macos)
            echo "Operating System: macOS"
            if detect_apple_silicon; then
                CMAKE_ARGS="-DGGML_METAL=ON"
                BUILD_TYPE="Metal (Apple Silicon GPU acceleration)"
            else
                # Intel Mac - use Accelerate framework
                CMAKE_ARGS="-DGGML_ACCELERATE=ON"
                BUILD_TYPE="CPU with Accelerate framework"
            fi
            ;;

        linux)
            echo "Operating System: Linux"

            # Check for NVIDIA CUDA
            if detect_nvidia; then
                # Check CUDA installation
                if command -v nvcc &> /dev/null; then
                    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
                    echo -e "${GREEN}✓ CUDA detected (version $CUDA_VERSION)${NC}"
                    CMAKE_ARGS="-DGGML_CUDA=ON"
                    BUILD_TYPE="CUDA (NVIDIA GPU acceleration)"
                else
                    echo -e "${YELLOW}⚠ NVIDIA GPU found but CUDA not installed${NC}"
                    echo "  Install CUDA toolkit for GPU acceleration"
                    if detect_vulkan; then
                        CMAKE_ARGS="-DGGML_VULKAN=ON"
                        BUILD_TYPE="Vulkan GPU acceleration"
                    else
                        CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
                        BUILD_TYPE="CPU with BLAS acceleration"
                    fi
                fi
            # Check for AMD ROCm
            elif detect_amd; then
                CMAKE_ARGS="-DGGML_HIPBLAS=ON"
                BUILD_TYPE="HIP/ROCm (AMD GPU acceleration)"
            # Check for Vulkan
            elif detect_vulkan; then
                CMAKE_ARGS="-DGGML_VULKAN=ON"
                BUILD_TYPE="Vulkan GPU acceleration"
            # Fall back to BLAS or CPU
            else
                CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
                BUILD_TYPE="CPU with BLAS acceleration"
            fi
            ;;

        windows)
            echo "Operating System: Windows"
            echo -e "${YELLOW}Note: For Windows, use CMake GUI or Developer Command Prompt${NC}"
            CMAKE_ARGS="-DGGML_CUDA=ON"  # Assuming CUDA on Windows
            BUILD_TYPE="CUDA (adjust if needed)"
            ;;

        *)
            echo -e "${RED}Unknown operating system${NC}"
            echo "Defaulting to basic CPU build"
            CMAKE_ARGS=""
            BUILD_TYPE="CPU-only"
            ;;
    esac

    # Save configuration for future runs
    cat > "$CONFIG_FILE" << EOF
CMAKE_ARGS="$CMAKE_ARGS"
BUILD_TYPE="$BUILD_TYPE"
EOF
    echo -e "${GREEN}✓ Saved configuration to $CONFIG_FILE${NC}"
fi

echo ""
echo "================================================"
echo -e "Selected build configuration: ${GREEN}${BUILD_TYPE}${NC}"
echo -e "CMake arguments: ${GREEN}${CMAKE_ARGS}${NC}"
echo "================================================"
echo ""

# Prompt user for confirmation
read -p "Proceed with this configuration? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Build cancelled."
    exit 0
fi

# Clone repository into a temporary build directory
LLAMA_BUILD_DIR="$PROJECT_DIR/.llama_build"
if [ ! -d "$LLAMA_BUILD_DIR" ]; then
    echo "Cloning llama.cpp repository (shallow)..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_BUILD_DIR"
fi

cd "$LLAMA_BUILD_DIR"

echo ""
echo "Configuring build with CMake..."
echo ""

# Create build directory
mkdir -p "$LLAMA_BUILD_DIR/build"
cd "$LLAMA_BUILD_DIR/build"

# Configure with CMake
echo "Running: cmake .. $CMAKE_ARGS"
cmake "$LLAMA_BUILD_DIR" $CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

echo ""
echo "Building llama.cpp..."
echo ""

# Build with all available cores
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . --config Release -j $NUM_CORES

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    # Extract only what's needed and clean up the build directory
    echo "Extracting build artifacts..."
    LLAMA_DIR="$PROJECT_DIR/llama.cpp"
    mkdir -p "$LLAMA_DIR/models"
    if [ ! -d "$PROJECT_DIR/adapters" ]; then
        mkdir -p "$PROJECT_DIR/adapters"
    fi

    # Copy the server binary
    BINARY_SRC=""
    if [ -f "$LLAMA_BUILD_DIR/build/bin/llama-server" ]; then
        BINARY_SRC="$LLAMA_BUILD_DIR/build/bin/llama-server"
    elif [ -f "$LLAMA_BUILD_DIR/build/bin/Release/llama-server" ]; then
        BINARY_SRC="$LLAMA_BUILD_DIR/build/bin/Release/llama-server"
    fi

    if [ -n "$BINARY_SRC" ]; then
        cp "$BINARY_SRC" "$LLAMA_DIR/llama-server"
        chmod +x "$LLAMA_DIR/llama-server"
        echo -e "${GREEN}✓ Copied llama-server binary${NC}"
    else
        echo -e "${RED}Could not find llama-server binary${NC}"
        exit 1
    fi

    # Copy the adapter conversion script
    cp "$LLAMA_BUILD_DIR/convert_lora_to_gguf.py" "$LLAMA_DIR/convert_lora_to_gguf.py"
    echo -e "${GREEN}✓ Copied convert_lora_to_gguf.py${NC}"

    # Remove the build directory
    echo "Cleaning up build directory..."
    rm -rf "$LLAMA_BUILD_DIR"
    echo -e "${GREEN}✓ Removed build directory${NC}"

    echo ""
    echo "Downloading model..."
    if [ ! -f "$LLAMA_DIR/models/llama-3.2-3b-instruct-q4_k_m.gguf" ]; then
        curl -L -o "$LLAMA_DIR/models/llama-3.2-3b-instruct-q4_k_m.gguf" \
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        echo -e "${GREEN}✓ Model downloaded${NC}"
    else
        echo -e "${GREEN}✓ Model already exists, skipping download${NC}"
    fi
    echo ""

else
    echo -e "${RED}Build failed!${NC}"
    echo "Please check the error messages above."
    exit 1
fi