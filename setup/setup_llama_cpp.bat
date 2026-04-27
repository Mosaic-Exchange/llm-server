@echo off
setlocal enabledelayedexpansion

echo === llama.cpp Automatic Setup Script (CMake - Windows) ===
echo.

:: Colors (using ANSI escape codes, supported in modern Windows terminals)
set "RED=[0;31m"
set "GREEN=[0;32m"
set "YELLOW=[1;33m"
set "NC=[0m"

:: Config file path
set "SCRIPT_DIR=%~dp0"
set "CONFIG_FILE=%SCRIPT_DIR%load.config"
set "PROJECT_DIR=%SCRIPT_DIR%.."

:: Check for CMake
where cmake >nul 2>exp1
if %ERRORLEVEL% neq 0 (
    echo %RED%x CMake not found%NC%
    echo Please install CMake and add it to your PATH.
    echo Download from: https://cmake.org/download/
    exit /b 1
) else (
    for /f "tokens=3" %%a in ('cmake --version ^| findstr "version"') do set "CMAKE_VERSION=%%a"
    echo %GREEN%v CMake found (version !CMAKE_VERSION!)%NC%
)

:: Check for cached configuration
if exist "%CONFIG_FILE%" (
    echo Found cached configuration at %CONFIG_FILE%
    for /f "usebackq delims=" %%a in ("%CONFIG_FILE%") do set "%%a"
    echo %GREEN%v Loaded CMAKE_ARGS and BUILD_TYPE from cache%NC%
) else (
    echo Detecting hardware configuration...
    echo.

    set "CMAKE_ARGS="
    set "BUILD_TYPE=CPU-only"

    :: Detect NVIDIA GPU
    where nvidia-smi >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        nvidia-smi >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            echo %GREEN%v NVIDIA GPU detected%NC%
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            
            :: Check for CUDA
            where nvcc >nul 2>&1
            if !ERRORLEVEL! equ 0 (
                for /f "tokens=*" %%a in ('nvcc --version ^| findstr "release"') do echo %GREEN%v CUDA detected (%%a)%NC%
                set "CMAKE_ARGS=-DGGML_CUDA=ON"
                set "BUILD_TYPE=CUDA (NVIDIA GPU acceleration)"
            ) else (
                echo %YELLOW%^! NVIDIA GPU found but CUDA toolkit (nvcc) not found in PATH%NC%
                echo Install CUDA toolkit for GPU acceleration.
                
                :: Try Vulkan as fallback
                where vulkaninfo >nul 2>&1
                if !ERRORLEVEL! equ 0 (
                    set "CMAKE_ARGS=-DGGML_VULKAN=ON"
                    set "BUILD_TYPE=Vulkan GPU acceleration"
                )
            )
        )
    ) else (
        :: Check for Vulkan if no NVIDIA
        where vulkaninfo >nul 2>&1
        if %ERRORLEVEL% equ 0 (
            echo %GREEN%v Vulkan support detected%NC%
            set "CMAKE_ARGS=-DGGML_VULKAN=ON"
            set "BUILD_TYPE=Vulkan GPU acceleration"
        )
    )

    :: Save configuration
    (
        echo CMAKE_ARGS="!CMAKE_ARGS!"
        echo BUILD_TYPE="!BUILD_TYPE!"
    ) > "%CONFIG_FILE%"
    echo %GREEN%v Saved configuration to %CONFIG_FILE%%NC%
)

echo.
echo ================================================
echo Selected build configuration: %GREEN%%BUILD_TYPE%%NC%
echo CMake arguments: %GREEN%%CMAKE_ARGS%%NC%
echo ================================================
echo.

set /p "CONFIRM=Proceed with this configuration? (y/n) "
if /i "%CONFIRM%" neq "y" (
    echo Build cancelled.
    exit /b 0
)

:: Setup build directories
if not defined PROJECT_DIR set "PROJECT_DIR=."
set "LLAMA_BUILD_DIR=%PROJECT_DIR%\.llama_build"

if not exist "%LLAMA_BUILD_DIR%" (
    echo Cloning llama.cpp repository (shallow)...
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "%LLAMA_BUILD_DIR%"
)

if not exist "%LLAMA_BUILD_DIR%\build" mkdir "%LLAMA_BUILD_DIR%\build"
cd /d "%LLAMA_BUILD_DIR%\build"

echo.
echo Configuring build with CMake...
echo.

:: Clean CMAKE_ARGS of quotes for execution
set "CLEAN_CMAKE_ARGS=%CMAKE_ARGS:"=%"
cmake "%LLAMA_BUILD_DIR%" %CLEAN_CMAKE_ARGS% -DBUILD_SHARED_LIBS=OFF

if %ERRORLEVEL% neq 0 (
    echo %RED%CMake configuration failed^!%NC%
    exit /b 1
)

echo.
echo Building llama.cpp...
echo.

:: Build using all cores (using NUMBER_OF_PROCESSORS env var)
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% equ 0 (
    echo.
    echo %GREEN%================================================%NC%
    echo %GREEN%Build successful^!%NC%
    echo %GREEN%================================================%NC%
    echo.
    
    echo Extracting build artifacts...
    set "LLAMA_DIR=%PROJECT_DIR%\llama.cpp"
    if not exist "%LLAMA_DIR%\models" mkdir "%LLAMA_DIR%\models"
    if not exist "%PROJECT_DIR%\adapters" mkdir "%PROJECT_DIR%\adapters"

    :: Copy the server binary (check multiple possible locations)
    set "BINARY_SRC="
    if exist "%LLAMA_BUILD_DIR%\build\bin\Release\llama-server.exe" (
        set "BINARY_SRC=%LLAMA_BUILD_DIR%\build\bin\Release\llama-server.exe"
    ) else if exist "%LLAMA_BUILD_DIR%\build\bin\llama-server.exe" (
        set "BINARY_SRC=%LLAMA_BUILD_DIR%\build\bin\llama-server.exe"
    )

    if defined BINARY_SRC (
        copy /y "!BINARY_SRC!" "%LLAMA_DIR%\llama-server.exe"
        echo %GREEN%v Copied llama-server binary%NC%
    ) else (
        echo %RED%Could not find llama-server binary%NC%
        exit /b 1
    )

    :: Copy the adapter conversion script
    copy /y "%LLAMA_BUILD_DIR%\convert_lora_to_gguf.py" "%LLAMA_DIR%\convert_lora_to_gguf.py"
    echo %GREEN%v Copied convert_lora_to_gguf.py%NC%

    :: Clean up
    echo Cleaning up build directory...
    cd /d "%PROJECT_DIR%"
    rmdir /s /q "%LLAMA_BUILD_DIR%"
    echo %GREEN%v Removed build directory%NC%

    echo.
    echo Downloading model...
    if not exist "%LLAMA_DIR%\models\llama-3.2-3b-instruct-q4_k_m.gguf" (
        curl -L -o "%LLAMA_DIR%\models\llama-3.2-3b-instruct-q4_k_m.gguf" ^
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        echo %GREEN%v Model downloaded%NC%
    ) else (
        echo %GREEN%v Model already exists, skipping download%NC%
    )
    echo.
) else (
    echo %RED%Build failed^!%NC%
    echo Please check the error messages above.
    exit /b 1
)

endlocal
