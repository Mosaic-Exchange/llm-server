# This file is for defining custom errors
# this is mainly to avoid the problem of flat errors that are uninformative
# This is primarily for internal use as the middleware server will be returning the error message in the format
# detailed there

# File path does not exist --> For when people want to register an adapter but the file path isn't there
# Uses FileNotFoundError

"""class AdapterPathNotFound(Exception):
    def __init__(self, adapter_path: str):
        self.message = f"Adapter path '{adapter_path}' was not found in the adapters folder."
        super().__init__(self.message)"""
from pathlib import Path

class AdapterUploadError(Exception):
    def __init__(self, adapter_path: Path, description: str):
        self.message = f"Adapter upload from '{adapter_path}' failed: {description}."
        super().__init__(self.message)

class InvalidSafetensorsUploadError(AdapterUploadError):
    def __init__(self, filepath: Path, description: str):
        self.message = f"The .safetensors style upload from '{filepath}' was invalid: {description}."
        super().__init__(filepath, self.message)

class InvalidGGUFUploadError(AdapterUploadError):
    def __init__(self, filepath: Path, description: str):
        self.message = f"The .gguf file at '{filepath}' was invalid: {description}."
        super().__init__(filepath, self.message)


# Adapter does not exist --> for when people request the usage of an adapter that does not exist

# Raised when an adapter ID passed the registry check but is missing from LlamaClient's known list
# — indicates an internal state inconsistency rather than a caller error
class AdapterStateError(Exception):
    pass

# Raised when the llama-server process cannot be reached (connection refused, process crashed, etc.)
class BackendUnavailableError(Exception):
    pass

# Raised specifically when _wait_for_server() exhausts its timeout; subclasses BackendUnavailableError
# so callers can catch at either granularity
class BackendReloadTimeoutError(BackendUnavailableError):
    pass

# Raised when the llama-server returns a non-2xx response on a chat completion request
class InferenceError(Exception):
    pass

# Raised when the llama-server response does not match the expected schema (e.g. missing "choices")
class MalformedBackendResponseError(Exception):
    pass