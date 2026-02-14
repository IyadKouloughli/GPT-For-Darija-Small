import argparse

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import darija_tokenizer


# Function to perform text generation using the trained Darija GPT model
def generate_text(prompt, model_path="checkpoints/model_final.pt", max_length=100, num_return_sequences=3):
    # Detect the best available hardware (GPU/CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model architecture with the standard config (must match training)
    # 50,000 is our standardized Darija vocabulary size
    model = GPT(GPTConfig(vocab_size=50000))

    # Load the trained model weights from the specified file path
    state_dict = torch.load(model_path, map_location=device)
    
    # Pre-process the state_dict to handle keys if the model was saved using torch.compile
    # (Removes the '_orig_mod.' prefix added by the compiler)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_state_dict[k.replace("_orig_mod.", "")] = v
    
    # Load the cleaned weights into the model
    model.load_state_dict(cleaned_state_dict)

    # Move model to the correct device (GPU/CPU) and set to evaluation mode
    model.to(device)
    model.eval()

    # Convert the input Darija prompt into a sequence of token IDs
    tokens = darija_tokenizer.encode(prompt)
    # Create a PyTorch tensor from the token IDs
    tokens = torch.tensor(tokens, dtype=torch.long)

    # Prepare for batch generation: repeat the prompt tensor for each requested sequence
    # Adds a batch dimension (e.g., from [T] to [1, T]) then repeats to [num_return_sequences, T]
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    # Move tokens to the processing device
    x = tokens.to(device)

    # Main autoregressive generation loop
    while x.size(1) < max_length:
        # Disable gradient calculation for efficiency during inference
        with torch.no_grad():
            # Get the model's predictions for the current sequence
            logits, _ = model(x)
            # Focus only on the last predicted token's distribution
            logits = logits[:, -1, :]
            # Convert raw logits into probabilities using Softmax
            probs = F.softmax(logits, dim=-1)
            # Select the top 50 most likely candidates (Top-K sampling)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # Sample one token from the Top-K candidates for each sequence in the batch
            ix = torch.multinomial(topk_probs, 1)
            # Retrieve the actual token ID corresponding to the selected index
            xcol = torch.gather(topk_indices, -1, ix)
            # Append the newly predicted token to our sequence
            x = torch.cat((x, xcol), dim=1)

    # Decode the final sequences back into human-readable Darija text
    generated_texts = []
    for i in range(num_return_sequences):
        # Extract individual token IDs for this sequence
        tokens = x[i, :max_length].tolist()
        # Decode token IDs using our Darija tokenizer
        decoded = darija_tokenizer.decode(tokens)
        # Store the decoded text
        generated_texts.append(decoded)
        
    # Return the list of generated strings
    return generated_texts


# Entry point for the inference script
def main():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate text using Darija-GPT")
    # --prompt: The text to start the generation from (in Darija)
    parser.add_argument(
        "--prompt", type=str, required=True, help="Input prompt text in Darija"
    )
    # --model_path: Where to find the saved .pt model file
    parser.add_argument(
        "--model_path", type=str, default="checkpoints/model_final.pt",
        help="Path to the saved model file (default: checkpoints/model_final.pt)"
    )
    # --max_length: The total number of tokens to generate
    parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum length of generated text"
    )
    # --num_return_sequences: How many different sentences to generate
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=3,
        help="Number of sequences to generate",
    )
    
    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Call the generation logic with the parsed arguments
    generated_texts = generate_text(
        args.prompt, args.model_path, args.max_length, args.num_return_sequences
    )
    
    # Print the results in a clean format to the console
    print("\n" + "=" * 60)
    for i, text in enumerate(generated_texts):
        print(f"\n--- Sequence {i+1} ---")
        print(text)
    print("\n" + "=" * 60)


# Run the script if called directly
if __name__ == "__main__":
    main()
