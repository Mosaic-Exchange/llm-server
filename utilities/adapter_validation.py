# This file is going to define an object that performs all validation and transformation (and rejection / cleanup)
# operations on the uploaded adapter files.
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Any
from gguf.gguf_reader import GGUFReader
from safetensors import safe_open

from .constants import (
    LLAMA_MODEL_PATH,
    LLAMA_ADAPTERS_DIR,
    LLAMA_CONVERT_SCRIPT,
    LLAMA_BASE_CONFIG,
    EXPECTED_ARCH,
    LLAMA_VALID_MODULES
)

from .backend_errors import (
    InvalidGGUFUploadError,
    AdapterUploadError,
    InvalidSafetensorsUploadError
)


# Its responsibilities are to return either true or false --> True if the adapter is (after the function processes)
# suitable to register with the LlamaClient and false otherwise (in which case there should be no trace)

# Positive Cases
# PC1 --> A valid .gguf adapter file is placed into the filepath
# PC2 --> A valid .gguf adapter file is placed into the filepath (plus extra files, e.g. system prompt, etc)
# PC3 --> A valid .safetensor PEFT version of adapters is placed into the filepath
# PC4 --> Any of PC1-PC3 plus some detritus files

# Negative Cases
# NC1 --> The specified filepath does not exist
# NC2 --> A .gguf file that does not fit the model was placed in the filepath
# NC3 --> A non-PEFT .safetensor file was placed into the specified filepath
# NC4 --> A valid .safetensor file was placed into the specified filepath without config information
# NC5 --> An incompatible set of otherwise valid .safetensor files were placed in the specified filepath
# NC6 --> Random files in the specified filepath (No adapter)

def validate_adapter(
        adapter_path: Path
) -> Optional[bool]:

    # Check if specified filepath exists (NC1)
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

    # Though 'sharded' LoRA adapeters (LoRA adapters that are so big they need to be split across multiple weight files)
    # are possible, we do not consider them here. This is because
    #   i. multiple weight files probably is an indication that this is a model, not an adapter (much more common)
    #   ii. we are running locally, loading an adapter that big would kill us on RAM consumption
    #   iii. Adding an adapter that huge onto a tiny 3B model like ours is an absurd use case
    if len(safetensors_files) > 1:
        raise AdapterUploadError(adapter_path, "Sharded adapters (multiple .safetensors files) are not supported.")
    if len(safetensors_files) == 1:
        return _handle_safetensors_case(adapter_path, safetensors_files[0])

    # Case #3: There is neither a .gguf nor is there a .safetensor file in the directory
    _garbage_collect(
        adapter_path=adapter_path,
        gc_list=None,
        trash=None
    )
    raise AdapterUploadError(adapter_path, "No .gguf or .safetensors file found in the adapter directory.")

def _get_files_by_extension(dir_path: Path, extension: str) -> List[Path]:
    return [
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix == extension
    ]

# This function is all about checking if the .gguf file that we found in the uploaded directory
# is actually what we want (a compatible file). Note that this strategy does not GUARANTEE anything,
# it is just a reasonable set of barriers to overcome.
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

    # Normalize filename: the .gguf file must be named after its parent directory (e.g. ella/ella.gguf)
    # so the rest of the system can locate it predictably. Rename if it doesn't already match.
    expected_name = dir_path.name + ".gguf"
    if gguf_path.name != expected_name:
        gguf_path.rename(dir_path / expected_name)

    # Do final clean up
    return _final_clean_up(adapter_dir=dir_path)

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

def _handle_safetensors_case(dir_path: Path, weight_file: Path) -> Optional[bool]:
    # CHECK 1: Requisite files
    # To be a valid PEFT style LoRA adapter we need an adapter_config.json file along with the weight file(s)
    # This file is pretty universally named adapter_config.json, so it's safe for us to assume that that is it's name.
    # Moreover, if that is not its name it is a reasonable indication something is wrong.
    config_path = dir_path / "adapter_config.json"
    if not config_path.exists():
        raise InvalidSafetensorsUploadError(dir_path, "Missing required file: adapter_config.json")

    # CHECK 2: The weight file is a valid safetensors file and actually contains LoRA weights.
    # We use the library to parse the header (no tensor data is loaded into memory).
    # Tensor names follow the PEFT pattern:
    #   base_model.model...{module}.lora_A.weight
    #   base_model.model...{module}.lora_B.weight
    # The presence of lora_A/lora_B in the tensor names is strong evidence of LoRA
    try:
        with safe_open(str(weight_file), framework="numpy") as f:
            tensor_names = list(f.keys())
    except Exception as e:
        # If the library can't parse it its an imposter .safetensors file (or at the least corrupted and not usable)
        raise InvalidSafetensorsUploadError(weight_file, f"Could not parse file: {e}") from e

    lora_a_keys = [k for k in tensor_names if "lora_A" in k]
    lora_b_keys = [k for k in tensor_names if "lora_B" in k]
    if not lora_a_keys or not lora_b_keys:
        raise InvalidSafetensorsUploadError(
            weight_file,
            "No lora_A/B tensors found: this doesn't appear to be a LoRA adapter.",
        )

    # CHECK #3: Check module names
    # Extract the module names actually present in the weights (e.g. q_proj, v_proj) and
    # validate them against the known Llama architecture - more reliable than trusting the config.
    # Tensor names look like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    modules_in_weights = set()
    for key in lora_a_keys: # Background knowledge note: lora_a and lora_b are always going to be the same wrt to layer they cover, its a 1 2 punch
        parts = key.split(".")
        lora_idx = parts.index("lora_A")
        if lora_idx > 0:
            modules_in_weights.add(parts[lora_idx - 1])

    # By looking at the layer names we can do a module check on the safetensor
    unknown = modules_in_weights - LLAMA_VALID_MODULES
    if unknown:
        raise InvalidSafetensorsUploadError(
            weight_file,
            f"Weights target modules not present in Llama architecture: {', '.join(sorted(unknown))}",
        )

    # CHECK 4: Config is valid JSON and contains all required PEFT fields
    # Check4.1 - Is the .json valid?
    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise InvalidSafetensorsUploadError(config_path, f"adapter_config.json is not valid JSON: {e}") from e

    # Check4.2 - Does the .json have the necessary fields
    required_fields = {"r", "lora_alpha", "target_modules", "base_model_name_or_path"}
    missing = required_fields - set(config.keys())
    if missing:
        raise InvalidSafetensorsUploadError(
            config_path,
            f"adapter_config.json is missing required fields: {', '.join(sorted(missing))}",
        )

    # Check4.3 - Does the config have an invalid rank parameter or alpha val
    rank = config.get("r")
    if not isinstance(rank, int) or rank <= 0:
        raise InvalidSafetensorsUploadError(config_path, f"Invalid LoRA rank 'r': {rank!r}. Must be a positive integer.")

    # alpha=0 means the adapter's contribution is scaled to zero - it has no effect on output
    alpha = config.get("lora_alpha")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise InvalidSafetensorsUploadError(config_path, f"Invalid lora_alpha: {alpha!r}. Must be a positive number.")

    # CHECK 5: Rank consistency
    # A lora_A tensor has shape [r, in_features]. We load one sample tensor and confirm its first
    # dimension matches the rank declared in adapter_config.json. A mismatch means the config and
    # the weights are out of sync, meaning a likely corrupt or mismatched upload.
    # Note: this is a partial geometry check only. Full geometric compatibility (i.e. whether
    # in_features aligns with the base model's projection dimensions) cannot be verified here
    # without loading the model itself (too expensive, rather handle the crash).
    try:
        with safe_open(str(weight_file), framework="numpy") as f:
            sample = f.get_tensor(lora_a_keys[0])
        if sample.shape[0] != rank:
            raise InvalidSafetensorsUploadError(
                weight_file,
                f"Declared rank r={rank} does not match actual tensor rank {sample.shape[0]}.",
            )
    except InvalidSafetensorsUploadError:
        raise
    except Exception:
        pass  # if shape check fails for any other reason, let conversion catch it

    # CHECK 6: The adapter is compatible with the base model
    # base_model_name_or_path is set by the trainer and encodes what model the adapter was trained on
    base_model = str(config.get("base_model_name_or_path", ""))
    if "llama" not in base_model.lower(): # TODO: Do non-instruct adapters work with instruct?
        raise InvalidSafetensorsUploadError(
            config_path,
            f"Adapter base model '{base_model}' does not appear to be a Llama model.",
        )

    # Then if all checks get passed we can do the conversion to .gguf format
    convert_adapter(dir_path)

    # Do final clean up
    return _final_clean_up(adapter_dir=dir_path)

# Most adapters are stored (usually on HuggingFace) in .safetensor format. However, the llama.cpp backend that we
# are running locally operates on a format called .gguf. Since we want to make our application accessible to a
# wider array of people, we choose to accept .safetensor formatted LoRA adapter files. This means that we have
# to convert said files. Luckily, the llama.cpp open source project contains a script that does this conversion.
# So in this function we call that script (and properly parametrize it) so that the adapter can be used.
# The base config provides the model architecture metadata the script needs to correctly shape the output weights.
def convert_adapter(adapter_dir: Path) -> Path:
    outfile = adapter_dir / f"{adapter_dir.name}.gguf"
    cmd = [
        sys.executable,
        str(LLAMA_CONVERT_SCRIPT),
        "--base", str(LLAMA_BASE_CONFIG),
        "--outfile", str(outfile),
        "--outtype", "q8_0",
        str(adapter_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return outfile

# Uploads are allowed to submit parameter suggestion requests (through a .json) and system prompt requests
# through a .txt file. We expect this to be submitted in the form of a text field / a form, and to have the
# files shaped by us. Therefore, we (perhaps naively) expect that there will be no issues with these files.
# TODO: Is this naive?
# But, we still do need to do some cleanup in the case that there is any detritus in the folder
def _final_clean_up(adapter_dir: Path) -> bool:
    keep_list = [adapter_dir / f"{adapter_dir.name}.gguf"]

    # We do check if they are empty, but that is it
    param_suggestion_path = adapter_dir / f"parameter_suggestions.json"
    system_prompt_path = adapter_dir / f"system_prompt.txt"

    # Check it exists, is a file, and is more than 0 bytes
    if param_suggestion_path.exists() and \
        param_suggestion_path.is_file() and \
        param_suggestion_path.stat().st_size > 0:

        # Then try and open it (and check if it has any suggestions even)
        try:
            with param_suggestion_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data not in ({}, []):
                keep_list.append(param_suggestion_path)
        except (json.JSONDecodeError, OSError):
            pass

    if system_prompt_path.exists() and \
        system_prompt_path.is_file() and \
        system_prompt_path.stat().st_size > 0 and \
        system_prompt_path.suffix.lower() == ".txt":

        try:
            with system_prompt_path.open("r", encoding="utf-8") as f:
                if bool(f.read().strip()):
                    keep_list.append(system_prompt_path)
        except OSError:
            pass


    return _garbage_collect(adapter_dir, keep_list, False)

def _garbage_collect(
    adapter_path: Path,
    gc_list: Optional[List[Path]],
    trash: Optional[bool],  # True: delete listed, False: delete everything NOT listed
) -> bool:
    resolved_adapter = adapter_path.resolve()

    # Always delete entire directory if list is None or empty
    if gc_list is None or len(gc_list) == 0:
        shutil.rmtree(resolved_adapter)
        return True

    names_set = {p.resolve() for p in gc_list}

    # Determine candidates
    if trash:
        # delete only listed paths
        candidates = list(names_set)
    else:
        # delete everything except listed paths
        candidates = list(resolved_adapter.iterdir())

    for target in candidates:
        try:
            resolved_target = target.resolve()

            # Ensure target stays inside adapter_path
            if resolved_target != resolved_adapter and resolved_adapter not in resolved_target.parents:
                return False

            # Decide deletion behavior
            should_delete = resolved_target in names_set if trash else resolved_target not in names_set

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
