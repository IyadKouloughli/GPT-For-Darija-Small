from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Determine the device to run the model on (prefer GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The GPTConfig class is a data class that holds the configuration parameters for the GPT-2 model
# These parameters define the size and complexity of the Darija-GPT model
@dataclass
class GPTConfig:
    block_size: int = 1024  # The maximum context length (number of tokens) the model can handle
    vocab_size: int = 50000  # The total number of unique tokens in the Darija vocabulary
    n_layer: int = 12  # The number of transformer blocks (decoder layers) in the model
    n_head: int = 12  # The number of attention heads in each transformer block
    n_embd: int = 768  # The dimensionality of the token embeddings and hidden states


# The MLP class defines a multi-layer perceptron (MLP) module for the GPT-2 model
# This is the "feed-forward" part of the transformer block
class MLP(nn.Module):
    # The __init__ method initializes the MLP module with the given configuration
    def __init__(self, config):
        super().__init__()
        # First linear layer expands the embedding dimension by a factor of 4
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU activation function provides non-linearity (approximate="tanh" is standard for GPT-2)
        self.gelu = nn.GELU(approximate="tanh")
        # Second linear layer projects the 4x expansion back to the original embedding dimension
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # Custom attribute to identify this as a residual projection for special initialization
        self.c_proj.NANOGPT_SCALE_INIT = 1

    # The forward method defines the forward pass of the MLP module
    def forward(self, x):
        # Apply the first linear expansion
        x = self.c_fc(x)
        # Apply the GELU non-linear activation
        x = self.gelu(x)
        # Project back to the embedding dimension
        x = self.c_proj(x)
        # Return the processed tensor
        return x


# The CausalSelfAttention class defines the multi-head self-attention mechanism
# "Causal" means tokens can only attend to previous tokens in the sequence
class CausalSelfAttention(nn.Module):
    # The __init__ method initializes the attention module
    def __init__(self, config):
        super().__init__()
        # Ensure the embedding size is cleanly divisible by the number of heads
        assert config.n_embd % config.n_head == 0
        # Linear layer to compute Query, Key, and Value vectors for all heads at once
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Linear layer for the output projection after attention
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Custom attribute for special initialization scaling
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # Store configuration as local attributes for convenience
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    # The forward pass for the attention mechanism
    def forward(self, x):
        # B: Batch size, T: Sequence length, C: Embedding dimension
        B, T, C = x.size()
        # Compute query, key, and value vectors (result shape: B, T, 3*C)
        qkv = self.c_attn(x)
        # Split the result into separate Q, K, and V tensors (each B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape and transpose for multi-head attention: (B, T, C) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Efficiently compute attention weights and weighted values using Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Concatenate headers back together: (B, n_head, T, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Apply the final output projection
        y = self.c_proj(y)
        # Return the attention output
        return y


# The DecoderBlock class defines one "layer" of the transformer
class DecoderBlock(nn.Module):
    # Each block contains self-attention followed by a feed-forward network
    def __init__(self, config):
        super().__init__()
        # Layer normalization before the attention mechanism
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # The multi-head self-attention module
        self.attn = CausalSelfAttention(config)
        # Layer normalization before the feed-forward network
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # The multi-layer perceptron (feed-forward) module
        self.mlp = MLP(config)

    # Forward pass through the decoder block with residual connections
    def forward(self, x):
        # Attention with residual connection (Pre-Norm architecture)
        x = x + self.attn(self.ln_1(x))
        # MLP with residual connection (Pre-Norm architecture)
        x = x + self.mlp(self.ln_2(x))
        # Return the block's output
        return x


# The main GPT class defining the model architecture for Darija text generation
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Store the configuration
        self.config = config

        # ModuleDict to organize the transformer components
        self.transformer = nn.ModuleDict(
            {
                # wte: Weight Token Embedding (maps token IDs to vectors)
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                # wpe: Weight Positional Embedding (maps positions to vectors)
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                # h: List of decoder blocks (the "brain" of the model)
                "h": nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
                # ln_f: Final Layer Normalization before the output layer
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )

        # The language modeling head: projects embeddings back to vocabulary size for prediction
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: shared weights between inputs and outputs for better performance/smaller size
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all model weights using a custom scheme
        self.apply(self._init_weights)

    # Private method to initialize weights according to GPT-2 standards
    def _init_weights(self, module):
        # Initialize linear layers
        if isinstance(module, nn.Linear):
            std = 0.02
            # Special scaling for residual layers to maintain variance
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # Initialize bias to zero if it exists
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # Initialize embedding layers
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Forward pass of the entire GPT model
    def forward(self, idx, target=None):
        # B: Batch size, T: Sequence length (number of input tokens)
        B, T = idx.size()
        # Verify that the input sequence doesn't exceed the configured block size
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of size {T}, block size is only {self.config.block_size}"
        
        # Create a range of indices for position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        # Lookup positional embeddings for each index in the sequence
        pos_embed = self.transformer.wpe(pos)
        # Lookup token embeddings for each token ID in the batch
        tok_embed = self.transformer.wte(idx)
        # Combine token and position embeddings (input to transformer blocks)
        x = tok_embed + pos_embed

        # Sequentially pass the hidden states through each transformer block
        for block in self.transformer.h:
            x = block(x)

        # Apply final layer normalization
        x = self.transformer.ln_f(x)
        
        # Project hidden states to vocabulary logits (predicting the next token)
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided (used during training)
        loss = None
        if target is not None:
            # Cross entropy loss comparing predicted logits to actual target tokens
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            
        # Return predicted logits and the calculated loss
        return logits, loss
