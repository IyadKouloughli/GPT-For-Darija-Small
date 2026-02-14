import os
import sys

# Add parent directory to path to allow importing model and tokenizer from the core codebase
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model import GPT, GPTConfig
from tokenizer import darija_tokenizer

# Initialize FastAPI application with a Darija-focused title
app = FastAPI(title="Darija-GPT Interface")

# Enable Cross-Origin Resource Sharing (CORS) to allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Detect and set the hardware device (GPU/CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to the trained model weights
MODEL_PATH = "checkpoints/model_final.pt"

# Logic to find a valid model file if the default 'model_final.pt' is missing
if not os.path.exists(MODEL_PATH):
    # Check for any .pt files in the checkpoints directory
    if os.path.exists("checkpoints"):
        pt_files = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
        if pt_files:
            # Use the first available .pt file as a fallback
            MODEL_PATH = os.path.join("checkpoints", pt_files[0])
        else:
            MODEL_PATH = None
    else:
        MODEL_PATH = None

# Global model variable
model = None

# Attempt to load the model into memory
if MODEL_PATH:
    try:
        # Initialize the model architecture with 50,000 vocab size for Darija
        model = GPT(GPTConfig(vocab_size=50000))
        # Load the saved state dict
        state_dict = torch.load(MODEL_PATH, map_location=device)
        # Clean up keys if the model was saved with torch.compile
        cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        # Load weights into the model instance
        model.load_state_dict(cleaned_state_dict)
        # Move model to the detected device (GPU/CPU)
        model.to(device)
        # Set to evaluation mode for inference
        model.eval()
        print(f"Darija model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading Darija model: {e}")

# Data structure for incoming generation requests
class GenerateRequest(BaseModel):
    prompt: str  # The Darija text to start generation from
    max_length: int = 100  # Total length of the generated sequence
    num_sequences: int = 1  # Number of variations to return

# API endpoint to handle text generation
@app.post("/generate")
async def generate(request: GenerateRequest):
    # Check if the model was loaded correctly before processing
    if model is None:
        raise HTTPException(status_code=500, detail="Darija model not loaded. Check checkpoints folder.")
    
    try:
        # Convert the Darija input text into tokens
        tokens = darija_tokenizer.encode(request.prompt)
        # Create a batch of tokens for parallel generation
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(request.num_sequences, 1)
        # Move tokens to the compute device
        x = tokens.to(device)

        # Autoregressive generation loop: predict tokens one by one
        while x.size(1) < request.max_length:
            with torch.no_grad():
                # Get raw model outputs (logits)
                logits, _ = model(x)
                # Select only the distribution for the final token
                logits = logits[:, -1, :]
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # Sample from Top-50 candidates
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                # Append the new token to the sequence
                x = torch.cat((x, xcol), dim=1)

        # Convert generated token IDs back into human-readable Darija text
        results = []
        for i in range(request.num_sequences):
            decoded = darija_tokenizer.decode(x[i].tolist())
            results.append(decoded)
        
        # Return the generated list of Darija strings
        return {"generated_texts": results}
    except Exception as e:
        # Return a server error if generation fails
        raise HTTPException(status_code=500, detail=str(e))

# Mount the 'frontend' directory to serve the HTML/JS/CSS user interface
app.mount("/", StaticFiles(directory="web_app/frontend", html=True), name="frontend")

# Run the backend server if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
