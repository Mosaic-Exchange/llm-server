from typing import Optional


class AdapterManager:
    # manage lifecycle of lora adapters

    def __init__(self, title, version):
        self.title = title
        self.version = version

    def convert_to_gguf(self, adapter_path):
        # calls convert_adapter_to_gguf.py
        pass

    def add_adapter(self, adapter_path):
        # validate / register new adapter
        pass

    def delete_adapter(self, adapter_id):
        # remove adapter from registry and unmount 
        pass 

    def load_adapter_cache(self):
        # loads adapter registry from disk, returns JSON
        pass

    def validate_adapter(self, adapter_path: str) -> bool:
        # check adapter is right file format and compatible w/ base model
        pass
