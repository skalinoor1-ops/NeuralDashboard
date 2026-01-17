from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import uvicorn

# CONFIGURATION
MODEL_PATH = r"D:\neural.gguf"
API_KEY = "NR_GH33S8GE65"

print(f"--- INITIALIZING BRAIN ON RTX 4090 ---")
print(f"--- LOADING: {MODEL_PATH} ---")

try:
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048)
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.post("/v1/chat")
    async def chat(request: Request):
        auth = request.headers.get("Authorization")
        if auth != f"Bearer {API_KEY}":
            return {"response": "ACCESS DENIED: Invalid API Key"}
        
        data = await request.json()
        output = llm(f"Q: {data['prompt']} A:", max_tokens=200, stop=["Q:"])
        return {"response": output["choices"][0]["text"]}

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=7860)
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    input("Press Enter to close...")
