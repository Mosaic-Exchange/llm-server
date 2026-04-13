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

class InvalidGGUFUploadError(Exception):
    def __init__(self, filepath: Path, description: str):
        self.message = f"The .gguf file at '{filepath}' was invalid: {description}."
        super().__init__(self.message)


# Adapter does not exist --> for when people request the usage of an adapter that does not exist