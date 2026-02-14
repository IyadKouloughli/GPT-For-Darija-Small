# 🇩🇿 Darija-GPT: Generative AI for Algerian Dialect

**Darija-GPT** is a project dedicated to building and training a Decoder-only Transformer (GPT-2 architecture) from scratch, optimized specifically for **Algerian Darija**. Unlike standard Arabic models, Darija-GPT is trained on a custom dataset of colloquial speech, capturing the unique linguistic nuances and conversational patterns of the Algerian dialect.

---

## Overview

Algerian Darija is a rich, spoken dialect with limited formal written resources. This project leverages the **GPT-2 Small (124M Parameters)** architecture to create a model capable of generating coherent and meaningful Darija text.

### Key Features:
- **Custom BPE Tokenizer**: A specialized Byte-Pair Encoding tokenizer with a 50,000-word vocabulary tailored for Darija.
- **Modern Transformer Architecture**: Uses Flash Attention, Causal Self-Attention, and Weight Tying for efficient training and inference.
- **End-to-End Pipeline**: A complete workflow from dataset acquisition to a premium web-based chat interface.

---

## Architecture

The model follows the GPT-2 Small specifications:
- **Layers**: 12 Transformer Blocks
- **Attention Heads**: 12
- **Embedding Dimension**: 768
- **Context Window**: 1,024 Tokens
- **Vocab Size**: 50,000 Tokens
- **Parameters**: ~124 Million

---

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ and a CUDA-compatible GPU (recommended) for training.

```bash
# Clone the repository
git clone https://github.com/your-username/GPT-For-Darija-Small.git
cd GPT-For-Darija-Small

# Install dependencies
pip install -r requirements.txt
```

### 2. The Training Pipeline

Follow these steps in order to prepare and train your model:

#### Step 1: Download and Prepare Data
Acquire the Algerian Darija dataset from Hugging Face and split it into training (90%) and validation (10%) sets.
```bash
python download_data.py
```
*Outputs: `data/train.txt` and `data/val.txt`*

#### Step 2: Train the Tokenizer
Train a custom Byte-Pair Encoder on the raw Darija text to handle the dialect's unique morphology.
```bash
python dataloader.py
```
*Outputs: `GPT2_darija_tokenizer/` directory.*

#### Step 3: Train the Model
Start the training process. The script includes Cosine Learning Rate decay, AdamW optimization, and early stopping.
```bash
python train.py
```
*Checkpoints will be saved in the `checkpoints/` folder.*

---

## Usage and Interface

### Command Line Inference
Generate text directly from your terminal:
```bash
python inference.py --prompt "السلام عليكم" --max_length 100
```

### Web Application
Experience the model through a modern, responsive web interface built with FastAPI and Vanilla CSS.
```bash
# Start the backend server
python web_app/app.py
```
Once running, visit **`http://localhost:8000`** in your browser.

---

## Project Structure

```text
├── data/               # Training and validation text files
├── checkpoints/        # Saved model weights (.pt)
├── web_app/            # FastAPI backend and Frontend assets
├── model.py            # GPT architecture implementation
├── tokenizer.py        # Tokenizer training logic
├── dataloader.py       # Custom data batching for Transformer
├── train.py            # Primary training loop
├── inference.py        # Text generation script
└── settings.py         # Global configuration & hyperparameters
```

---

## Credits & References
- **Dataset**: [ayoubkirouane/Algerian-Darija](https://huggingface.co/datasets/ayoubkirouane/Algerian-Darija)
- **Base Architecture**: Based on Andrej Karpathy's nanoGPT and OpenAI's GPT-2 research.

---
*Created with ❤️ for the Algerian Developer Community.*
