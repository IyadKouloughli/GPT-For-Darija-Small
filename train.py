import argparse
import math
import os
import time
import warnings

import torch

from dataloader import DataLoader
from model import GPT, GPTConfig

# ── Suppress known harmless warnings ────────────────────────────────────────
# Ignore warnings when token sequence length exceeds the model's context window
# This is expected behavior as the data is chunked into blocks
warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
# Ignore warnings about the renamed CUDA allocation configuration variable
warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF.*")

# ── Environment config ──────────────────────────────────────────────────────
# Use 'expandable_segments' to help manage GPU memory more efficiently
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Learning rate and training schedule hyperparameters
max_lr = 6e-4  # Maximum learning rate (at peak of warmup)
min_lr = max_lr * 0.1  # Minimum learning rate (after cosine decay)
warmup_steps = 10  # Number of steps to linearly increase LR from 0 to max_lr
max_steps = 4000  # Total number of training iterations
save_interval = 500  # Frequency of saving model checkpoints
eval_interval = 500  # Frequency of evaluating on the validation set
patience_limit = 3   # Stop training early if validation doesn't improve for this many evals
val_fraction = 0.1   # Default percentage of data for validation (if no val file)

# Function to calculate the learning rate for a given step using a cosine decay schedule with warmup
def get_lr(step):
    # Linear warmup phase
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # After max steps, stay at the minimum learning rate
    if step > max_steps:
        return min_lr
    # Calculate how far we are through the decay phase (0 to 1)
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    # Standard Cosine Decay formula
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# Automatically detect if a CUDA-enabled GPU is available for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Main training function
def train():
    # Initialize the GPT model; vocab_size should match the Darija tokenizer's capacity
    model = GPT(GPTConfig(vocab_size=50000))

    # Training configuration constants
    B = 4  # Mini-batch size
    T = 128  # Sequence length (tokens per sample)
    train_file = "data/train.txt"  # Path to the training text file
    val_file = "data/val.txt"  # Path to the validation text file

    # Initialize data loaders based on file availability
    if os.path.exists(val_file) and os.path.getsize(val_file) > 0:
        # If a dedicated validation file exists, use it
        data_loader = DataLoader(B=B, T=T, file_path=train_file)
        val_loader = DataLoader(B=B, T=T, file_path=val_file)
    else:
        # Fallback: Split the main training file into training and validation sets
        with open(train_file, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if len(lines) < 2:
            raise ValueError("Not enough lines to split for validation.")
            
        # Calculate split index based on the val_fraction
        split_idx = max(1, int(len(lines) * (1.0 - val_fraction)))
        train_text = "\n".join(lines[:split_idx])
        val_text = "\n".join(lines[split_idx:])
        
        # Create loaders using raw text strings
        data_loader = DataLoader(B=B, T=T, text=train_text)
        val_loader = DataLoader(B=B, T=T, text=val_text)

    # Ensure the checkpoints directory exists for saving models
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set model to evaluation mode for setup and move to device
    model.eval()
    model.to(device)

    # Enable TensorFloat-32 (TF32) on Ampere GPUs for faster floating-point math
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Attempt to compile the model for optimized performance (requires PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"torch.compile unavailable or failed: {e}. Continuing without compile.")
        
    # Initialize the AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8
    )

    # Helper function to evaluate the model's loss on the validation dataset
    def evaluate(model, val_loader):
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            # Iterate through validation batches
            for bx, by in val_loader.iterate_batches():
                bx = bx.to(device)
                by = by.to(device)
                # Forward pass without gradient tracking
                _, loss = model(bx, by)
                total_loss += loss.item()
                count += 1
        # Calculate and return the average validation loss
        return total_loss / max(1, count)

    # Training state variables
    best_val_loss = float("inf")
    no_improve_evals = 0

    # Start the main training loop
    for step in range(max_steps):
        # Timestamp for performance measurement
        t0 = time.time()

        # Fetch the next training batch (inputs x and targets y)
        x, y = data_loader.next_batch()

        # Move training tensors to the target device
        x = x.to(device)
        y = y.to(device)

        # Clear accumulated gradients from the previous step
        optimizer.zero_grad()

        # Execute the forward pass through the model
        logits, loss = model(x, y)

        # Execute the backward pass to calculate gradients
        loss.backward()

        # Clip gradients to prevent them from becoming too large (exploding gradients)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the learning rate according to our schedule
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Perform the optimization step (update model weights)
        optimizer.step()

        # Synchronize CPU and GPU to get accurate timing metrics
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        # End step timing and calculate metrics
        t1 = time.time()
        dt = (t1 - t0) * 1000 # Time in milliseconds
        tokens_per_sec = (data_loader.B * data_loader.T) / (t1 - t0) # Throughput

        # Log training progress to the console
        print(
            f"step {step}, loss: {loss.item():.4f}, dt: {dt:.2f}ms, norm: {norm:.4f}, tok/sec: {tokens_per_sec:.2f}"
        )

        # Periodic evaluation and model checkpointing
        if (step + 1) % eval_interval == 0:
            # Run evaluation on validation data
            val_loss = evaluate(model, val_loader)
            print(f"[eval] step {step+1}, val_loss: {val_loss:.6f}")

            # Save a regular checkpoint
            if (step + 1) % save_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"model_step_{step + 1}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Model checkpoint saved at {ckpt_path}")

            # Logic to track the best model and initiate early stopping
            if val_loss < best_val_loss - 1e-12:
                # If validation loss improved, save this as the new best model
                best_val_loss = val_loss
                no_improve_evals = 0
                best_name = f"best_val_step_{step+1}_val_{val_loss:.4f}.pt"
                best_path = os.path.join(ckpt_dir, best_name)
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved: {best_path}")
            else:
                # If no improvement, increment the patience counter
                no_improve_evals += 1
                print(f"No improvement count: {no_improve_evals}/{patience_limit}")

            # Stop training if the patience limit is reached without improvement
            if no_improve_evals >= patience_limit:
                print(f"Early stopping triggered at step {step+1}")
                break

    # Save the final model state after training is complete or interrupted
    final_path = os.path.join(ckpt_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved at {final_path}")


# Execution entry point
if __name__ == "__main__":
    train()
