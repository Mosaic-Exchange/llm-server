@echo off
setlocal enabledelayedexpansion

:: Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%" || (
    echo Error: Could not cd to script directory %SCRIPT_DIR%
    echo Error: Could not cd to script directory %SCRIPT_DIR% >&2
    exit /b 1
)

set "PIPENV_CMD="
set "PYTHON_CMD="
set "VENV_BIN_DIR=.venv-pipenv\Scripts"

:: Check if the pre-existing venv's pipenv is already installed (fastest path)
if exist "!VENV_BIN_DIR!\pipenv.exe" (
    set "PIPENV_CMD=!VENV_BIN_DIR!\pipenv"
) else (
    :: Check if pipenv is on the system PATH
    where pipenv >nul 2>nul
    if !errorlevel! == 0 (
        set "PIPENV_CMD=pipenv"
    ) else (
        :: Pipenv not found anywhere; need Python to bootstrap it
        where python >nul 2>nul
        if !errorlevel! == 0 (
            set "PYTHON_CMD=python"
        ) else (
            echo Error: Python not found. Please install Python 3 to run this server.
            echo Error: Python not found. Please install Python 3 to run this server. >&2
            exit /b 1
        )

        echo Pipenv not found on PATH. Creating a virtual environment to install pipenv...
        "!PYTHON_CMD!" -m venv .venv-pipenv || (
            echo Error: Failed to create virtualenv for pipenv
            echo Error: Failed to create virtualenv for pipenv >&2
            exit /b 1
        )

        "!VENV_BIN_DIR!\pip" install pipenv || (
            echo Error: Failed to install pipenv into virtual environment
            echo Error: Failed to install pipenv into virtual environment >&2
            exit /b 1
        )

        set "PIPENV_CMD=!VENV_BIN_DIR!\pipenv"
    )
)

:: Run pipenv install from the script's directory
:: --skip-lock lets pipenv resolve for the current platform instead of using
:: the Linux-generated Pipfile.lock (which pins Linux-only CUDA packages).
echo Running !PIPENV_CMD! install...
call !PIPENV_CMD! install --skip-lock || (
    echo Error: '!PIPENV_CMD! install' failed in %SCRIPT_DIR%
    echo Error: '!PIPENV_CMD! install' failed in %SCRIPT_DIR% >&2
    exit /b 1
)

:: Run the server from within the setup\ directory
cd setup || (
    echo Error: Could not cd into setup\ directory
    echo Error: Could not cd into setup\ directory >&2
    exit /b 1
)

echo Starting server...
call !PIPENV_CMD! run python -m uvicorn middleware_server:app %* || (
    echo Error: Failed to start uvicorn server
    echo Error: Failed to start uvicorn server >&2
    exit /b 1
)

endlocal
