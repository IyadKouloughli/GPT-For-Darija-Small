import torch
from tokenizer import darija_tokenizer

# The DataLoader class defines a data loader for the GPT-2 model specialized for Darija
class DataLoader:
    # The __init__ method initializes the data loader with the batch size (B) and the sequence length (T)
    # It accepts either a file path via `file_path` or a raw `text` string.
    def __init__(self, B, T, file_path: str = None, text: str = None):
        # Set the batch size (B)
        self.B = B
        # Set the sequence length (T)
        self.T = T

        # Obtain raw text from either provided text or file
        if text is not None:
            # Use raw text if provided directly
            raw_text = text
        else:
            # If no file path is provided, default to the training data path
            if file_path is None:
                file_path = "data/train.txt"
            
            # Attempt to open and read the Darija dataset file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except FileNotFoundError:
                # Raise an error if the specified file does not exist
                raise FileNotFoundError(f"{file_path} not found. Please ensure training/validation data exists.")

        # Check if the raw text is empty or contains only whitespace
        if not raw_text or not raw_text.strip():
            raise ValueError(f"{file_path or 'provided text'} is empty. Please provide training/validation text.")

        # Tokenize the Darija text using the specialized darija_tokenizer
        # This converts the text into a list of integers (tokens)
        tokens = darija_tokenizer.encode(raw_text)

        # Convert the list of tokens to a PyTorch tensor for efficient processing on CPU/GPU
        self.tokens = torch.tensor(tokens)

        # Ensure that tokenization actually produced something
        if self.tokens.numel() == 0:
            raise ValueError("Tokenization produced no tokens. Check the tokenizer and input text/file.")

        # Ensure there's enough data for at least one full batch (B*T tokens plus one for the target sequence)
        if len(self.tokens) < (self.B * self.T + 1):
            raise ValueError(
                f"Not enough tokens ({len(self.tokens)}) for one batch (B={self.B}, T={self.T}). Reduce batch/sequence size or add more data."
            )

        # Initialize the current position to 0; this tracks whereabouts in the token stream we are
        self.current_position = 0

    # The next_batch method returns the next batch of training data
    def next_batch(self):
        # Load local copies of batch size (B) and sequence length (T)
        B, T = self.B, self.T

        # Get a chunk of tokens starting from the current position
        # We take B*T + 1 tokens because the target (y) is shifted by one position from the input (x)
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        # x consists of the first B*T tokens, reshaped into (B, T)
        # These are the input sequences for the model
        x = (buf[:-1]).view(B, T)
        
        # y consists of the next tokens (shifted by 1), also reshaped into (B, T)
        # These are the tokens the model is expected to predict
        y = (buf[1:]).view(B, T)

        # Advance the current position by the number of tokens processed in this batch
        self.current_position += T * B

        # If the next batch would exceed the available tokens, wrap back to the beginning of the dataset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        # Return the input (x) and target (y) tensors
        return x, y

    # Generator method to yield batches sequentially for validation or evaluation
    def iterate_batches(self):
        """Yield (x, y) batches sequentially for the entire dataset without wrap-around."""
        # Load local copies of batch size and sequence length
        B, T = self.B, self.T
        # Calculate total tokens in the dataset
        total_tokens = len(self.tokens)
        # Calculate tokens per batch
        batch_size_tokens = B * T
        # Calculate total number of full batches available
        num_full_batches = total_tokens // batch_size_tokens
        
        # Iterate through the number of full batches
        for i in range(num_full_batches):
            # Calculate the start index for the current batch
            start = i * batch_size_tokens
            # Get the chunk of tokens for this batch
            buf = self.tokens[start : start + batch_size_tokens + 1]
            # Reshape inputs (x) and targets (y)
            x = (buf[:-1]).view(B, T)
            y = (buf[1:]).view(B, T)
            # Yield the pair of tensors
            yield x, y

    # Calculate the total number of batches available in the dataset
    def num_batches(self):
        # Local copies of B and T
        B, T = self.B, self.T
        # Integer division of total tokens by tokens per batch
        return len(self.tokens) // (B * T)
