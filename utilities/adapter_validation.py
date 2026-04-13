# This file is going to define an object that performs all validation and transformation (and rejection / cleanup)
# operations on the uploaded adapter files.
import os
import shutil
from pathlib import Path
from typing import Optional, List, Any
from gguf.gguf_reader import GGUFReader

from constants import (
    LLAMA_MODEL_PATH,
    LLAMA_ADAPTERS_DIR,
    EXPECTED_ARCH
)

from backend_errors import (
    InvalidGGUFUploadError,
    AdapterUploadError
)


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
        adapter_path: str
) -> Optional[bool]:

    # Check if specified filepath exists (NC1)
    adapter_path = Path(LLAMA_ADAPTERS_DIR) / adapter_path
    if not adapter_path.exists() or not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter path '{adapter_path}' was not found.")

    # Case #1: There is a .gguf file in the specified directory
    # Collect all files ending with .gguf
    gguf_files = _get_files_by_extension(adapter_path, ".gguf")
    if len(gguf_files) == 1:
        return _handle_gguf_case(adapter_path, gguf_files[0])
    elif len(gguf_files) > 1:
        raise AdapterUploadError(adapter_path, f"Multiple gguf files: {gguf_files}")

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

# This function is all about checking if the .gguf file that we found in the uploaded directory
# is actually what we want (a compatible file)
def _handle_gguf_case(dir_path: Path, filename: Path) -> Optional[bool]:
    gguf_path = dir_path / filename

    # CHECK 1: File Size
    # The first check we do is on the file size. If the file size is greater than the size of the
    # model itself, then we can assume something is wrong (probably that the indicated file is not
    # an adapter, but maybe a full model). And even if it doesn't, it would be too memory heavy
    if gguf_path.stat().st_size > Path(LLAMA_MODEL_PATH).stat().st_size:
        raise InvalidGGUFUploadError(gguf_path, "GGUF file too large.")

    # CHECK #2: Magic Numbers
    # .gguf files have the a little signature at the beginning. So we can read the first few bytes
    # of the file. If the bytes don't align with what they should be for a .gguf file then we know
    # we have an imposter file
    with open(gguf_path, "rb") as f:
        magic = f.read(4)

    if magic != b"GGUF":
        raise InvalidGGUFUploadError(gguf_path, "Missing Magic Number.")

    # CHECK #3: Try Parsing
    # llama.cpp defines a GGUFReader to try and read gguf files. If this reader fails, that means that
    # mounting it would too
    try:
        reader = GGUFReader(str(gguf_path))
    except Exception as e:
        raise InvalidGGUFUploadError(gguf_path, f"GGUFReader() failed to parse: {e}") from e

    # CHECK #4: Metadata evaluation
    # GGUF files contain plenty of metadata. At this point in the program flow we know we have a valid
    # gguf file. Now we need to find out if that file is a LoRA adapter, and if that adapter is compatible
    arch = _field_value(reader, "general.architecture")
    if _stringify(arch).lower() != EXPECTED_ARCH:
        raise InvalidGGUFUploadError(gguf_path, "Invalid architecture.")

    adapter_type = _field_value(reader, "adapter.type")
    if _stringify(adapter_type).lower() != "lora":
        raise InvalidGGUFUploadError(gguf_path, "Invalid adapter type.")


    # CHECK #5: Read tensors of the gguf
    # Using the reader we can collect the tensors and see if they match up with what we would
    # expect. This means that we are looking for common lora keywords (e.g. lora) that would be
    # associated with LoRA weights. We also check if there are too many weights that DON'T have
    # that lora association, as that too could point to it being a full model.
    names = [t.name for t in reader.tensors]

    lora_like = [
        n for n in names
        if ".lora_a" in n.lower()
           or ".lora_b" in n.lower()
           or "lora_a" in n.lower()
           or "lora_b" in n.lower()
    ]

    if len(lora_like) == 0:
        raise InvalidGGUFUploadError(gguf_path, "No lora like tensors.")

    regular_weight_like = [
        n for n in names
        if n.endswith(".weight")
           and "lora" not in n.lower()
    ]

    if len(regular_weight_like) >= 100:
        raise InvalidGGUFUploadError(gguf_path, "Too many regular weights. Looks model like.")

    return True

def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    return str(value)

def _field_value(reader: GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.get_field(key)
    if field is None:
        return default
    try:
        return field.contents()
    except Exception:
        return default

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

if __name__ == "__main__":
    print("hello")
