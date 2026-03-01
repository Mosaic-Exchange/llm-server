from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Mosaic AI API", version="0.1.0")

class GenerationCreateRequest(BaseModel):
    message: str
    request_id: str
    max_tokens: Optional[str] = None
    adapter_id: Optional[str] = None

class AdapterCreateRequest(BaseModel):
    adapter_filename: str
    request_id: str

class AdapterDeleteRequest(BaseModel):
    adapter_id: str

class ErrorInner(BaseModel):
    type: str
    code: str
    message: str
    err_id: str

class ErrorEnvelope(BaseModel):
    error: ErrorInner

class GenerationObject(BaseModel):
    request_id: str
    object: str = "generation"
    output: str

class AdapterObject(BaseModel):
    adapter_id: str
    adapter_filename: str
    object: str = "adapter"

class AdapterListObject(BaseModel):
    object: str = "list"
    data: list[AdapterObject]

@app.post("/generation", response_model=GenerationObject)
def create_generation(request: GenerationCreateRequest):
    # stub: accepts generation request and returns placeholder 
    # will route through LlamaClient -> llama.cpp server

    return GenerationObject(
        request_id=request.request_id,
        output="stub: placeholder generation response"
    )


@app.post("/adapter", response_model=AdapterObject)
def create_adapter(request: AdapterCreateRequest):
    # stub: registers new adapter file
    # will validate/convert file via AdapterManager

    return AdapterObject(
        adapter_id="stub-adapter-id",
        adapter_filename=request.adapter_filename
    )


@app.delete("/adapter", response_model=dict)
def delete_adapter(request: AdapterDeleteRequest):
    # stub: removes an adapter by ID
    # will call AdapterManager.delete_adapter()

    return {"deleted": True, "adapter_id": request.adapter_id}


@app.get("/adapters", response_model=AdapterListObject)
def list_adapters():
    # stub: returns empty list
    # will read from the AdapterManager cache and return adapter list

    return AdapterListObject(data=[])