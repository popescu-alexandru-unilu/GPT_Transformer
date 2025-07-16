import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sentencepiece as spm
from decoder import MyDecoder
from inference import generate

# --- Pydantic models for validated request/response shapes ---
class PromptRequest(BaseModel):
    prompt: str

class GenerationResponse(BaseModel):
    generated_text: str

# --- Model & Application Configuration ---
MODEL_PATH = './models/GPT_model_best.pth'
SPM_MODEL_PATH = './spm_vocab_text8_32k.model'
SEQ_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 8
D_FF = 2048
NUM_HEADS = 8

# --- Initialize FastAPI App and Load Model on Startup ---
app = FastAPI(title="GPT Inference API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ML Service: Using device: {device}")

tokenizer = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)
model = MyDecoder(
    vocab_size=tokenizer.get_piece_size(), max_seq_length=SEQ_LENGTH, d_model=D_MODEL,
    num_layers=NUM_LAYERS, d_ff=D_FF, num_heads=NUM_HEADS
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("ML Service: Model loaded successfully.")

# --- API Endpoint Definition ---
@app.post("/generate", response_model=GenerationResponse)
async def handle_generation(request: PromptRequest):
    """Receives a prompt and returns model-generated text."""
    try:
        generated_text = generate(
            model=model, tokenizer=tokenizer, prompt=request.prompt,
            max_new_tokens=150, temperature=0.8, top_k=40,
            device=device, max_seq_length=SEQ_LENGTH
        )
        return GenerationResponse(generated_text=generated_text)
    except Exception as e:
        print(f"ML Service: Error during generation: {e}")
        raise HTTPException(status_code=500, detail="Internal error during text generation.")
