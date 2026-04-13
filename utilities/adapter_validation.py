# This file is going to define an object that performs all validation and transformation (and rejection / cleanup)
# operations on the uploaded adapter files.
import os
import shutil
from pathlib import Path
from typing import Optional, List
import glob


# Its responsibilities are to return either true or false --> True if the adapter is (after the function processes)
# suitable to register with the LlamaClient and false otherwise (in which case there should be no trace)

# Positive Cases
# PC1 --> A valid .gguf adapter file is placed into the filepath
# PC2 --> A valid .gguf adapter file is placed into the filepath (plus extra files, e.g. system prompt, etc)
# PC3 --> A valid .safetensor PEFT version of adapters is placed into the filepath
# PC4 --> A valid .gguf that is the wrong quantization
# PC5 --> A valid .safetensor that is the wrong quantization
# PC6 --> Any of PC1-PC3 plus some detritus files

# Negative Cases
# NC1 --> The specified filepath does not exist
# NC2 --> A .gguf file that does not fit the model was placed in the filepath
# NC3 --> A non-PEFT .safetensor file was placed into the specified filepath
# NC4 --> A valid .safetensor file was placed into the specified filepath without config information
# NC5 --> An incompatible set of otherwise valid .safetensor files were placed in the specified filepath
# NC6 --> Random files in the specified filepath (No adapter)

def validate_adapter(
        adapter_dir: Path,
        adapter_path: str
) -> Optional[bool]:

    # Check if specified filepath exists (NC1)
    adapter_path = Path(adapter_dir) / adapter_path
    if not adapter_path.exists() or not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter path '{adapter_path}' was not found.")

    # Case #1: There is a .gguf file in the specified directory
    # Collect all files ending with .gguf
    gguf_files = _get_files_by_extension(adapter_path, ".gguf")
    if len(gguf_files) != 0:
        return _handle_gguf_case() # TODO: What if someone uploads an invalid .gguf with a valid safetensor?

    # Case #2: There is a .safetensor file in the specified directory
    safetensors_files = _get_files_by_extension(adapter_path, ".safetensors")
    if len(safetensors_files) != 0:
        return _handle_safetensors_case()

    # Case #3: There is neither a .gguf nor is there a .safetensor file in the directory
    # In this case trash it all (we know there is at least a directory, empty or with garbage)
    _garbage_collect(
        adapter_path=adapter_path,
    )

    return False

def _get_files_by_extension(dir_path: Path, extension: str) -> List[Path]:
    return [
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix == extension
    ]

def _handle_gguf_case():
    return True

def _handle_safetensors_case():
    return True

def _garbage_collect(
    adapter_path: Path,
    gc_list: Optional[List[str]],
    trash: Optional[bool],  # True: delete listed, False: delete everything NOT listed
) -> bool:
    resolved_adapter = adapter_path.resolve()

    # Always delete entire directory if list is None or empty
    if gc_list is None or len(gc_list) == 0:
        shutil.rmtree(resolved_adapter)
        return True

    names_set = set(gc_list)

    # Determine candidates
    if trash:
        # delete only listed names
        candidates = [resolved_adapter / name for name in names_set]
    else:
        # delete everything except listed names
        candidates = list(resolved_adapter.iterdir())

    for target in candidates:
        try:
            resolved_target = target.resolve()

            # Ensure target stays inside adapter_path
            if resolved_target != resolved_adapter and resolved_adapter not in resolved_target.parents:
                return False

            name = target.name

            # Decide deletion behavior
            should_delete = name in names_set if trash else name not in names_set

            if not should_delete or not resolved_target.exists():
                continue

            if resolved_target.is_file() or resolved_target.is_symlink():
                resolved_target.unlink()
            elif resolved_target.is_dir():
                shutil.rmtree(resolved_target)

        except Exception:
            return False

    return True


