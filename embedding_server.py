from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch

app = FastAPI(title="BGE-M3 Embedding Server")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer('BAAI/bge-m3', device=device)

class EmbedRequest(BaseModel):
    texts: list[str]

@app.post('/embed')
async def embed(request: EmbedRequest):
    embeddings = model.encode(request.texts, device=device).tolist()
    return {'embeddings': embeddings}

@app.get('/health')
async def health():
    return {'status': 'ok', 'device': device}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)
