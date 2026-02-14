import pandas as pd
import os

# Ensure the 'data' directory exists to store the downloaded and processed Darija text
if not os.path.exists("data"):
    # Create the directory if it's missing
    os.makedirs("data")

# Dictionary mapping split names to their respective Hugging Face data file paths
splits = {'train': 'data/train-00000-of-00001.parquet', 'v1': 'data/v1-00000-of-00001.parquet'}

print("Downloading data from Hugging Face...")
# Read the v1 split of the Algerian Darija dataset directly from Hugging Face via the 'hf://' protocol
# This requirement was specifically requested by the user
df = pd.read_parquet("hf://datasets/ayoubkirouane/Algerian-Darija/" + splits["v1"])

# Extract the "Text" column, remove empty entries, ensure all data are strings, and convert to a Python list
text = df["Text"].dropna().astype(str).to_list()

# Split the cleaned data into 90% training and 10% validation sets
n = len(text)  # Total number of available samples
train_size = int(n * 0.9)  # 90% for training
train_data = text[:train_size]
val_data = text[train_size:]  # Remaining 10% for validation

# Log the dataset statistics to the console
print(f"Total samples: {n}")
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Save the training portion to 'data/train.txt' with UTF-8 encoding (required for Darija characters)
with open("data/train.txt", "w", encoding="utf-8") as f:
    # Join the list of strings with newlines
    f.write("\n".join(train_data))

# Save the validation portion to 'data/val.txt' with UTF-8 encoding
with open("data/val.txt", "w", encoding="utf-8") as f:
    # Join the list of strings with newlines
    f.write("\n".join(val_data))

# Notify the user that the process is finished
print("Data saved to data/train.txt and data/val.txt")
