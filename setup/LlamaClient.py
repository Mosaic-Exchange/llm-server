from typing import Optional


class LlamaClient:
    # class responsible for interacting w/ llama.cpp server + using AdapterManager

    def __init__(self, base_url, mount_adapters, initialize = True):
        self.base_url = base_url
        self.mount_adapters = mount_adapters
        self.initialized = False

        if initialize:
            self.initialize()

    def generate_chat(self, message: str, adapter_id: Optional[str] = None) -> str:
        # send chat query to llama.cpp server and return output response

        return "stub: placeholder chat response"

    def initialize(self) -> None:
        # initialize connection to llama.cpp server and set up AdapterManager
        
        self.initialized = True
        pass
