#!/bin/bash
set -euo pipefail

# Function to report error to both stdout and stderr
report_error() {
    echo "$1"
    echo "$1" >&2
}

# Get the directory where the script is located
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || { report_error "Error: Could not cd to script directory $SCRIPT_DIR"; exit 1; }

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

PIPENV_CMD=""
PYTHON_CMD=""

# Check if pipenv is on the path
if command_exists "pipenv"; then
    PIPENV_CMD="pipenv"
else
    # Pipenv not on path, check for python or python3
    if command_exists "python3"; then
        PYTHON_CMD="python3"
    elif command_exists "python"; then
        PYTHON_CMD="python"
    else
        report_error "Error: Python not found. Please install Python 3 to run this server."
        exit 1
    fi

    echo "Pipenv not found on PATH. Creating a virtual environment to install pipenv..."
    "$PYTHON_CMD" -m venv .venv-pipenv || { report_error "Error: Failed to create virtualenv for pipenv"; exit 1; }

    # Install pipenv within the virtualenv
    VENV_BIN_DIR="./.venv-pipenv/bin"
    if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "win32" || "${OSTYPE:-}" == "cygwin" ]]; then
        VENV_BIN_DIR="./.venv-pipenv/Scripts"
    fi

    "$VENV_BIN_DIR/pip" install pipenv || { report_error "Error: Failed to install pipenv into virtual environment"; exit 1; }
    
    PIPENV_CMD="$VENV_BIN_DIR/pipenv"
fi

# Run pipenv install from the script's directory
echo "Running $PIPENV_CMD install..."
$PIPENV_CMD install || { report_error "Error: '$PIPENV_CMD install' failed in $SCRIPT_DIR"; exit 1; }

# Run the server from within the setup/ directory
cd "setup" || { report_error "Error: Could not cd into setup/ directory"; exit 1; }

echo "Starting server..."
$PIPENV_CMD run python -m uvicorn middleware_server:app "$@" || { report_error "Error: Failed to start uvicorn server"; exit 1; }

