import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

import requests

from hardware import compute_max_loaded_adapters

logger = logging.getLogger(__name__)

ADAPTER_MEMORY_PATH = Path(__file__).parent.parent / "database" / "adapter_memory.json"


class LlamaClient:

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        server_binary: Optional[str] = None,
        model_path: Optional[str] = None,
        adapters_dir: Optional[str] = None,
        convert_script: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/") # http address of running llama.cpp back back end
        self._server_binary = server_binary # Path to llama.cpp server executable
        self._model_path = model_path # path to the base model's .gguf
        self._adapters_dir = Path(adapters_dir) if adapters_dir else None # path to adapter directory
        self._convert_script = Path(convert_script) if convert_script else None # path to conversion script
        self._server_process: Optional[subprocess.Popen] = None #
        self._port = urlparse(self.base_url).port or 8080 # port num. Always the same

        # The maximum number of adapters that can be loaded at a given time (e.g. |self._active_adapters|) is system
        # dependent. Thus we compute this dynamically at setup and load it in
        self.MAX_LOADED_ADAPTERS = compute_max_loaded_adapters(
            model_path=self._model_path,
            adapters_dir=self._adapters_dir,
        )

        # Known adapters are all the adapters that the system has registered with the AI Server
        # This means that they are all in the right format, the server knows about them, and they can be called upon
        self._known_adapters: List[str] = []

        # The active adapters are all the adapters that are on the actively loaded model. As the number of adapters that
        # can be known to the server is bounded by the systems disk and the number of active adapters is memory bound,
        # this list is either a subset of the known adapters (or equal to the known adapters). Adapters are evicted and
        # added to this list according to an LRU policy
        self._active_adapters: List[str] = []

        # Some adapters (e.g. DPO type adapters) are activated by special system prompts. Thus this must be taken into
        # account. This dictionary maps adapters to their system prompts, should they be needed
        self._system_prompts: Dict[str, str] = {}

        # Some adapters also have suggestions for parameter constraints (e.g. don't use this model with a temperature
        # below 0.8) that is not in line with the default parameter configurations for the base model. This dictionary
        # stores those reccomendations and makes adjustments should they be needed.
        self._parameter_suggestions: Dict[str, Dict[str, Any]] = {}

        # loads the records of what adapters should be loaded / are known from previous sessions should the server be
        # shut down between uses of the application. This loads from the database/adapter_memory.json file
        self._load_adapter_memory()

        self._ensure_server_running()

    # Just loads the memories for adapter information
    # This makes it so that the adapters that were previously verified (known) and loaded (active) can be restored
    # from the .json file between system restarts
    def _load_adapter_memory(self):
        if not ADAPTER_MEMORY_PATH.exists():
            return
        with open(ADAPTER_MEMORY_PATH, "r") as f:
            data = json.load(f)
        self._known_adapters = data.get("known_adapters", [])
        self._active_adapters = data.get("active_adapters", [])
        self._system_prompts = data.get("system_prompts", {})
        self._parameter_suggestions = data.get("parameter_suggestions", {})

    # Just in the same manner that the load adapter function loads things from the .json, the LlamaClient instance
    # is also responsible for saving that information to the aforemnetioned .json
    def _save_adapter_memory(self):
        ADAPTER_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ADAPTER_MEMORY_PATH, "w") as f:
            json.dump(
                {
                    "known_adapters": self._known_adapters,
                    "active_adapters": self._active_adapters,
                    "system_prompts": self._system_prompts,
                    "parameter_suggestions": self._parameter_suggestions,
                },
                f,
                indent=2,
            )

    # This function takes the (processed) adapter data (all associated files) and logs it as a known adapter
    # as a bonus this information is updated in the .json. Validation is NOT performed here.
    def register_adapter(
        self,
        filename: str,
        system_prompt: Optional[str] = None,
        parameter_suggestions: Optional[Dict[str, Any]] = None,
    ):
        if filename not in self._known_adapters:
            self._known_adapters.append(filename)
        if system_prompt is not None:
            self._system_prompts[filename] = system_prompt
        if parameter_suggestions is not None:
            self._parameter_suggestions[filename] = parameter_suggestions
        self._save_adapter_memory()

    # When adapters are deleted they need to be unregistered to prevent requests to non-existent adapters.
    # this means updating the class variables, but also updating the history .json to prevent mistakes / errors
    # on future startups.
    def unregister_adapter(self, filename: str):
        if filename in self._known_adapters:
            self._known_adapters.remove(filename)
        if filename in self._active_adapters:
            self._active_adapters.remove(filename)
        self._system_prompts.pop(filename, None)
        self._parameter_suggestions.pop(filename, None)
        self._save_adapter_memory()

    # Getter for the system prompt. Just for keeping encapsulation respected as much as possible between middleware
    # and the LlamaClient.py file
    def get_system_prompt(self, filename: str) -> Optional[str]:
        return self._system_prompts.get(filename)

    # Getter for the parameter suggestions. Just for keeping encapsulation respected as much as possible between middleware
    # and the LlamaClient.py file
    def get_parameter_suggestions(self, filename: str) -> Optional[Dict[str, Any]]:
        return self._parameter_suggestions.get(filename)

    # Most adapters are stored (usually on HuggingFace) in .safetensor format. However, the llama.cpp backend that we
    # are running locally operates on a format called .gguf. Since we want to make our application accessible to a
    # wider array of people, we choose to accept .safetensor formatted LoRA adapter files. This means that we have
    # to convert said files. Luckily, the llama.cpp open source project contains a script that does this conversion.
    # So in this function we call that script (and properly parametrize it) so that the adapter can be used.
    def convert_adapter(self, adapter_dir: Path) -> Path:
        base_config = Path(__file__).parent / "llama32_3b_config"
        outfile = adapter_dir / f"{adapter_dir.name}.gguf"

        cmd = [
            sys.executable,
            str(self._convert_script),
            "--base", str(base_config),
            "--outfile", str(outfile),
            "--outtype", "q8_0",
            str(adapter_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)

        return outfile

    # When you are making a query to a given adapter the adapter needs to not only be known but to be actively mounted
    # onto the base model instance. That is what this function ensures. Additionally the LRU policy for the adapters
    # that are considered active needs to be maintained; that happens here.
    def _ensure_adapter_active(self, filename: str) -> int:
        if filename not in self._known_adapters:
            raise ValueError(f"Adapter '{filename}' is not in the known adapters list.")

        # This part is for implementiing the LRU policy we mention in the constructor commentary / documentation
        # The 'if' updates the ordered
        if filename in self._active_adapters:
            self._active_adapters.remove(filename)
            self._active_adapters.append(filename)
            self._save_adapter_memory()
        # and the 'else' implements the eviction logic
        else:
            if len(self._active_adapters) >= self.MAX_LOADED_ADAPTERS:
                evicted = self._active_adapters.pop(0)
                logger.info(f"Evicting LRU adapter: {evicted}")
            self._active_adapters.append(filename)
            self._save_adapter_memory()
            self._reload_server()

        return self._active_adapters.index(filename)


    def _ensure_server_running(self):
        """
        This function serves to make sure that the background server is running.
        The application has the two servers, the middleware one and the llama_cpp foundation
        Before both had to be started individually, but this way a server user never has to touch
        that other server (completely invisible)
        """
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2)
            if r.status_code == 200:
                try:
                    loaded = requests.get(f"{self.base_url}/lora-adapters", timeout=2).json()
                    server_names = {Path(a["path"]).name for a in loaded}
                    if server_names != set(self._active_adapters):
                        logger.info("Server adapter state out of sync --> reloading.")
                        self._reload_server()
                except Exception:
                    pass
                return
        except Exception:
            pass

        if not self._server_binary or not self._model_path:
            raise RuntimeError(
                "llama-server is not running and server_binary/model_path were not provided to start it."
            )

        logger.info("llama-server not detected - starting it now.")
        self._reload_server()

    # This gets called a lot more than the name might imply. Every time the selection of active adapters changes, for
    # any reason, the server needs to be reloaded. Yes, we implement hotswapping between active adapters using scaling
    # factor modulations, but in order to change what adapters are active the whole model must be torn down and
    # reloaded. However, due to favorable caching background processes, this turns out to be a rather painless process.
    # In order to do the reload the server binary from llama.cpp is re-executed (with the appropriate 'active' adapter
    # list, as recorded in the instance variable).
    def _reload_server(self):
        if not self._server_binary or not self._model_path:
            raise RuntimeError(
                "server_binary and model_path must be provided to LlamaClient to support adapter reload."
            )

        if self._server_process and self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()

        cmd = [
            self._server_binary,
            "-m", self._model_path,
            "--host", "127.0.0.1",
            "--port", str(self._port),
        ]
        for filename in self._active_adapters:
            adapter_path = (self._adapters_dir / filename) if self._adapters_dir else Path(filename)
            cmd.extend(["--lora", str(adapter_path)])

        logger.info(f"Reloading llama-server: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._wait_for_server()

    # When starting up the server there is some downtime (that varies based on an individual's machine). Particularly
    # on the startup (at least for a MacBook Pro 2021 with the M1 Max Chip) startup can take ~10 seconds, though it is
    # longer on more lightweight machines. Thus we select a timeout of 120 seconds of waiting for that setup process
    # to complete beofre reporting an error.
    def _wait_for_server(self, timeout: int = 120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    adapters = requests.get(f"{self.base_url}/lora-adapters", timeout=2).json()
                    if len(adapters) == len(self._active_adapters):
                        logger.info("llama-server is ready.")
                        return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError(f"llama-server did not become ready within {timeout}s.")

    # Communicates directly with the llama.cpp llama-server to find out what adapters are currently active on top of the
    # base model. A wrapper to a llama.cpp server call essentially.
    def list_adapters(self) -> List[Dict]:
        response = requests.get(f"{self.base_url}/lora-adapters")
        response.raise_for_status()
        return response.json()

    # This function wraps a call to the llama.cpp llama-server with some additional logic that implements the
    # hotswappability that makes this project exciting. To get into that I am going to include a brief math explanation,
    # which can be skipped if desired. That section will be below:
    #
    # START OF BRIEF MATH
    # One way to think of LLM response generation is as taking an input, representing it numerically, and passing it
    # through a series of multiplications. And then, when it comes out the other end of all those multiplications it's
    # an answer to your input (obviously extremely simplified). A traditional finetune of a model will update the
    # coefficients of those multipliers that the input gets tumbled through, changing the answer that comes out the
    # other end. LoRA adapters are different. Rather than messing with the multipliers in between, LoRA adapters are
    # a few extra multiplications (few reffering to the fact that they are small in relation to the weights of the base
    # model). With our framework, we attach a whole selection (the active adapters list) onto the end of our base model.
    # However, for our purposes, we only want at most one adapter to influence the final output. The use adapter
    # essentially multiplies the effect of one adapter (the selected one) by 1, and the rest by zero. This means that
    # that one adapters multiplication will be applied to what comes out of the base model multiplication process,
    # whereas the rest will not.
    # END OF BRIEF MATH
    #
    # This function scales the adapters such that the desired one is applied to your input and the other active adapters
    # do not have any impact. It then makes the appropriate call to the llama-server and returns the response.
    def use_adapter(self, adapter_id: int, scale: float = 1.0) -> Dict:
        adapters_info = self.list_adapters()
        payload = [
            {"id": a["id"], "scale": scale if a["id"] == adapter_id else 0.0}
            for a in adapters_info
        ]
        response = requests.post(f"{self.base_url}/lora-adapters", json=payload)
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            response.raise_for_status()
        return response.json()

    # Much like use_adapter (see above for math explanation) but sets the effect of all loaded / active adapters to 0.
    # This means that you can still use the base model as if it had no adapters attached without having to reload the
    # model. In fact, you never need to reload the model to use the base model without any adapters. Same as the
    # use_adapter() function, this function also serves as a wrapper for a call to the llama-server from llama.cpp
    def use_base_only(self):
        adapters_info = self.list_adapters()
        if not adapters_info:
            return
        payload = [{"id": a["id"], "scale": 0.0} for a in adapters_info]
        response = requests.post(f"{self.base_url}/lora-adapters", json=payload)
        response.raise_for_status()

    # Gets adapter id by its filename (not as robust, potential filename overlaps, but so much semantically clearer)
    # TODO: Maybe detect for duplicate adapter file names at upload time. I'm just worried about people uploading the same adapter multiple times.
    def use_adapter_by_name(self, filename: str, scale: float = 1.0) -> int:
        # TODO: deprecate adapter_id in favor of this
        llama_id = self._ensure_adapter_active(filename)
        self.use_adapter(llama_id, scale)
        return llama_id

    # The base unit of actually chatting with the LLM. Formats the response with all the parameters that are configured
    # elsewhere, typically with the default parameters associated with the base model (the llama3b instruct). This
    # function can be called in either streaming or non-streaming mode. Mainly just formats arguments to the streaming
    # and non-streaming functions for chat specifically (no interaction with the llama-server back back end)
    #
    # Note: Though the application uses the streaming mode, replacing the non-streaming mode, we chose to not
    # deprecate the latter.
    def chat(
        self,
        message: str,
        adapter_filename: Optional[str] = None,
        adapter_scale: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 500,
        min_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ):
        if adapter_filename is not None:
            self.use_adapter_by_name(adapter_filename, adapter_scale)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if min_p is not None:
            payload["min_p"] = min_p

        if stream:
            return self._stream_chat(payload)
        return self._send_chat(payload)

    # Takes the message that was formatted in chat() and sends a properly formatted request to the llama-server back
    # back end. It then waits for a response and returns it to the caller. This is the non-streaming version, so the
    # output is returned in one big block.
    def _send_chat(self, payload: Dict) -> str:
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # Similarly to the non-streaming version _send_chat() this function takes the message formatted in the chat()
    # function and forwards it to the llama-server back back end. However, as this is the streaming version there are
    # some differences in that process making it less straightforward. First the stream flag is set to true in the
    # payload to the llama-server, and second the returned value must be read differently as it is being streamed
    # back to the caller rather than being returned in one big block. The section that reads the data is commented in
    # more detail for curious readers.
    def _stream_chat(self, payload: Dict) -> Iterator[str]:
        payload["stream"] = True
        response = requests.post(
            f"{self.base_url}/v1/chat/completions", json=payload, stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines(): # Read response body one line at a time
            if line: # skip blanks
                line = line.decode("utf-8") # decode from raw
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]": # Checking for flag to see when streaming is DONE
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get("choices"):
                            content = chunk["choices"][0].get("delta", {}).get("content", "") # Following OpenAI streaming convention
                            if content:
                                yield content # make a _stream_chat generator where each token fragment is yielded to caller when it comes in
                    except json.JSONDecodeError:
                        continue


if __name__ == "__main__":
    client = LlamaClient()

    print("loaded adapters:")
    adapters = client.list_adapters()
    print(json.dumps(adapters, indent=2))

    print("\nadapter by name:")
    response = client.chat(
        "Say 'I am using the adapter'",
        adapter_filename="structured_answer_adapter.gguf",
    )
    print(response)