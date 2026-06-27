---

language:

* en
  license: other
  pretty_name: GPT-2 From Scratch
  tags:
* gpt2
* transformer
* language-model
* pytorch
* text-generation

---

# GPT-2 From Scratch

This project is a personal learning project where I implemented a GPT-style language model from scratch using PyTorch.

The main goal of this project is not to build a large production-level model, but to understand how modern language models work internally, from tokenization and embeddings to attention, training, and text generation.

## What I Learned

Through this project, I learned the core ideas behind GPT and transformer-based language models:

* How text is converted into tokens using the GPT-2 tokenizer
* How token embeddings and positional embeddings represent input text
* How the attention mechanism works using Query, Key, and Value matrices
* Why causal masking is needed in autoregressive language models
* How multi-head attention allows the model to learn different relationships between tokens
* How transformer blocks are built using attention, feed-forward layers, layer normalization, and residual connections
* How a language model predicts the next token step by step
* How text generation works during inference
* How decoding methods such as greedy search, beam search, and temperature sampling affect the output
* Why temperature can make the model more creative by increasing randomness in token selection
* Why training AI models is expensive in terms of data, GPU memory, compute, and time

## Project Overview

The model follows the GPT-style decoder-only transformer architecture.

Main components:

```text
Input Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Token Embedding + Positional Embedding
   ↓
Transformer Blocks
   ↓
Final Layer Normalization
   ↓
Output Head
   ↓
Next Token Prediction
```

## Model Configuration

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True
}
```

## Text Generation

The model generates text autoregressively. This means it predicts one token at a time, then uses the generated token as part of the next input.

Example process:

```text
Prompt: "Question: What is AI?"
        ↓
Model predicts next token
        ↓
Append token to the prompt
        ↓
Repeat until max length or end token
```

## Decoding Methods

During generation, different decoding methods can change the style of the output.

### Greedy Search

Greedy search always selects the token with the highest probability.

It is simple and stable, but it can make the output repetitive or less creative.

### Beam Search

Beam search keeps multiple possible sequences at the same time and chooses the best overall sequence.

It can produce better structured outputs, but it is slower than greedy search.

### Temperature Sampling

Temperature controls how random the model is when choosing the next token.

* Low temperature: safer and more deterministic
* High temperature: more diverse and creative
* Too high temperature: output may become unstable or meaningless

This helped me understand why creativity in language models comes from sampling, not only from the model itself.

## Training Understanding

This project also helped me understand why training large AI models is expensive.

Training requires:

* A large amount of text data
* Many forward and backward passes
* Large GPU memory for activations and gradients
* Expensive matrix multiplications in attention layers
* Long training time, especially when model size and context length increase

The attention mechanism is powerful, but it becomes expensive because each token needs to compare with many other tokens in the sequence.

## Key Takeaways

This project helped me understand the foundation of modern LLMs.

The most important lessons were:

* GPT models are trained to predict the next token
* Attention allows the model to focus on relevant previous tokens
* Causal masking prevents the model from seeing future tokens
* Generation is a repeated next-token prediction process
* Search and sampling strategies strongly affect model output
* Training AI models is costly because of model size, data size, sequence length, and hardware requirements

## Reference

This project was inspired by the book:

**Building a Large Language Model From Scratch**
by Sebastian Raschka
