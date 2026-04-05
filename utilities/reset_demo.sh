#!/bin/bash

# Delete all subdirectories in adapters/
find /Users/danielarturi/Desktop/361_Group/llm-server/adapters -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# Delete adapter memory and load config
rm -f /Users/danielarturi/Desktop/361_Group/llm-server/database/adapter_memory.json
rm -f /Users/danielarturi/Desktop/361_Group/llm-server/setup/load.config

# Delete llama.cpp directory
rm -rf /Users/danielarturi/Desktop/361_Group/llm-server/llama.cpp
