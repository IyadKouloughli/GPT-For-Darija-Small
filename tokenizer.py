import torch
from transformers import AutoTokenizer

import settings

# Define a generator function to read Darija text data in batches from a file
# This is efficient for large datasets as it doesn't load everything into memory at once
def text_iterator(file_path, batch_size):
    # Open the dataset file with UTF-8 encoding to handle Darija characters correctly
    with open(file_path, encoding="utf-8") as file:
        batch = []  # Initialize an empty list to store the current batch of lines
        # Iterate over each line in the Darija dataset file
        for line in file:
            # Strip whitespace and add the line to the current batch
            batch.append(line.strip())
            # Check if the desired batch size has been reached
            if len(batch) == batch_size:
                # Yield the batch to the caller (e.g., the tokenizer trainer)
                yield batch
                # Clear the batch list to start collecting the next set of lines
                batch = []
        # If there are any remaining lines after the loop, yield them as the final batch
        if batch:
            yield batch

# Load the base GPT-2 tokenizer which provides the architecture for our Darija tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Train a specialized tokenizer for the Darija dialect using Byte-Pair Encoding (BPE)
# It uses the text_iterator to efficiently process the dataset according to the defined VOCAB_SIZE
darija_tokenizer = tokenizer.train_new_from_iterator(
    text_iterator(file_path=settings.DATASET_PATH, batch_size=settings.BATCH_SIZE), 
    vocab_size=settings.VOCAB_SIZE
)

# Save the trained Darija tokenizer files (vocab.json, merges.txt, etc.) for later use in training and inference
darija_tokenizer.save_pretrained("GPT2_darija_tokenizer")