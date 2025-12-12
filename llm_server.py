from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch

app = FastAPI(title="EXAONE-3.5-32B Server")

# Load model and tokenizer
print("Loading model...")
model_id = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"

# Check CUDA availability (NVIDIA DIGITS - Unified Memory)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# DIGITS unified memory - use device_map with max_memory hint
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_memory={0: "100GiB", "cpu": "50GiB"},
    offload_buffers=True,
)

device = next(model.parameters()).device
print(f"Model loaded! Device: {device}")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7

class CompletionRequest(BaseModel):
    model: str = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct", "object": "model"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # Build conversation
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }]
    }

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return {
        "id": "cmpl-1",
        "object": "text_completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "text": response_text,
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30000)
