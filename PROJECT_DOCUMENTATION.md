# GPT-2 from Scratch: Comprehensive Project Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Learning Source](#learning-source)
3. [Project Overview](#project-overview)
4. [Core Components](#core-components)
5. [The Attention Mechanism](#the-attention-mechanism)
6. [Transformer Architecture](#transformer-architecture)
7. [Text Generation Mechanism](#text-generation-mechanism)
8. [Training Costs and Computational Complexity](#training-costs-and-computational-complexity)
9. [Key Learnings](#key-learnings)

---

## Introduction

This project implements a GPT-2 (Generative Pre-trained Transformer-2) model from scratch using PyTorch. The implementation covers all essential components of a transformer-based language model, including tokenization, embedding layers, multi-head self-attention mechanisms, transformer blocks, and text generation pipelines.

The model is built incrementally, starting from fundamental concepts and progressing to a fully functional language model capable of generating coherent text sequences.

---

## Learning Source

**Book:** Building a Large Language Model from Scratch  
**Author:** Sebastian Raschka  
**Focus:** This comprehensive resource provides step-by-step guidance on constructing transformer models, understanding the mathematical foundations, and implementing efficient training procedures.

---

## Project Overview

### Model Configuration

The GPT-2 124M model is configured with the following hyperparameters:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,        # GPT-2 vocabulary size
    "context_length": 256,      # Maximum sequence length (tokens)
    "emb_dim": 768,             # Embedding dimension
    "n_heads": 12,              # Number of attention heads
    "n_layers": 12,             # Number of transformer blocks
    "drop_rate": 0.1,           # Dropout rate for regularization
    "qkv_bias": False           # Whether to use bias in Q, K, V projections
}
```

### Architecture Overview

The complete GPT model consists of:

1. **Token Embedding Layer**: Converts token IDs to embedding vectors
2. **Positional Embedding Layer**: Adds positional information to embeddings
3. **Embedding Dropout**: Regularization applied to combined embeddings
4. **12 Transformer Blocks**: Each containing multi-head attention and feed-forward networks
5. **Layer Normalization**: Stabilizes training
6. **Output Head**: Projects to vocabulary size for token prediction

---

## Core Components

### 1. Tokenization

The project uses **GPT-2 tokenizer** via the `tiktoken` library:

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode("Your text here")
```

**Key Points:**
- Converts raw text into integer token IDs
- GPT-2 uses Byte-Pair Encoding (BPE)
- Vocabulary contains 50,257 unique tokens
- Essential for bridging language and numerical computation

### 2. Embedding Layers

**Token Embedding:**
- Maps token IDs to 768-dimensional vectors
- Learned during training
- Each token gets a unique vector representation

**Positional Embedding:**
- Adds position information to tokens
- Allows the model to understand token order
- Without this, a sequence "ABC" would be identical to "CBA"
- Generated as learned embeddings based on position index

---

## The Attention Mechanism

### Understanding Attention

Attention is the core innovation that makes transformers powerful. It allows the model to dynamically focus on different parts of the input sequence when processing each token.

### Self-Attention Process

**Step 1: Linear Projections**

Each input sequence is projected into three representations:

$$Q = X \cdot W_Q$$
$$K = X \cdot W_K$$
$$V = X \cdot W_V$$

Where:
- $Q$ = Query matrix (what we're looking for)
- $K$ = Key matrix (what we can attend to)
- $V$ = Value matrix (the information to aggregate)
- $W_Q, W_K, W_V$ = Learnable weight matrices

**Step 2: Attention Scores Calculation**

Compute similarity between queries and keys:

$$\text{scores} = Q \cdot K^T / \sqrt{d_k}$$

Where $d_k$ is the dimension of keys (head dimension = 768/12 = 64 for GPT-2).

The scaling factor $1/\sqrt{d_k}$ prevents attention scores from becoming too large.

**Step 3: Causal Masking (Masked Self-Attention)**

This is crucial for language modeling. Each token can only attend to previous tokens and itself, not future tokens:

$$\text{mask} = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}$$

Applied before softmax:

$$\text{scores} = \text{scores} + \text{mask}$$

This ensures position $i$ cannot attend to positions $j$ where $j > i$.

**Step 4: Softmax Normalization**

Convert scores to attention weights:

$$\text{weights} = \text{softmax}(\text{scores}) = \frac{e^{\text{scores}}}{\sum e^{\text{scores}}}$$

Positions masked with $-\infty$ become $0$ after softmax, effectively preventing attention to them.

**Step 5: Context Vector Creation**

Weight values using attention weights:

$$\text{context} = \text{weights} \cdot V$$

Each token now has a context vector that combines information from all attended-to positions.

### Multi-Head Attention

Instead of a single attention operation, the model performs multiple attention operations in parallel:

1. Split the embedding dimension into multiple heads: $d_o = d_{emb} / n_{heads} = 768 / 12 = 64$
2. Each head learns to attend to different parts of the sequence
3. Concatenate all head outputs: $[head_1 \| head_2 \| ... \| head_{12}]$
4. Project back to original dimension through output layer

**Benefits:**
- Different heads capture different types of relationships
- Some heads may focus on syntax, others on semantics
- Provides diverse representation learning

### Implementation Code

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, n_tokens, d_in = x.shape
        
        # Linear projections
        queries = self.W_query(x)  # (b, n_tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Reshape for multi-head: (b, n_tokens, num_heads, head_dim)
        queries = queries.view(b, n_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, n_tokens, self.num_heads, self.head_dim)
        values = values.view(b, n_tokens, self.num_heads, self.head_dim)
        
        # Transpose: (b, num_heads, n_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Attention scores
        attn_scores = queries @ keys.transpose(2, 3)  # (b, num_heads, n_tokens, n_tokens)
        
        # Apply causal mask
        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # Softmax to get weights
        attn_weights = torch.softmax(
            attn_scores / self.head_dim**0.5,
            dim=-1
        )
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context_vec = attn_weights @ values  # (b, num_heads, n_tokens, head_dim)
        
        # Reshape back: (b, n_tokens, d_out)
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(b, n_tokens, -1)
        
        # Output projection
        context_vec = self.out_proj(context_vec)
        
        return context_vec
```

---

## Transformer Architecture

### Transformer Block Structure

Each transformer block combines three key components:

1. **Multi-Head Self-Attention**
2. **Feed-Forward Network (FFN)**
3. **Residual Connections with Layer Normalization**

### Block Implementation

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention sub-layer with residual connection
        shortcut = x
        x = self.norm1(x)  # Layer norm before attention (pre-norm)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection
        
        # Feed-forward sub-layer with residual connection
        shortcut = x
        x = self.norm2(x)  # Layer norm before FFN
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection
        
        return x
```

### Feed-Forward Network

Each transformer block contains a two-layer feed-forward network:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

The hidden dimension is typically 4 times the embedding dimension (4 × 768 = 3072 for GPT-2).

### Residual Connections

Residual connections (shortcuts) are critical:

$$\text{output} = \text{layer}(x) + x$$

**Benefits:**
- Enable training of very deep networks (12 layers)
- Gradient flow improvement during backpropagation
- Preserve information from previous layers

### Complete GPT Model

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Stack of transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        # Embedding
        tok_embeds = self.tok_emb(in_idx)  # (batch, seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # (batch, seq_len, emb_dim)
        x = self.drop_emb(x)
        
        # Transformer blocks
        x = self.trf_blocks(x)  # (batch, seq_len, emb_dim)
        
        # Output layer
        x = self.final_norm(x)
        logits = self.out_head(x)  # (batch, seq_len, vocab_size)
        
        return logits
```

### Model Size

Total parameters in GPT-2 124M:

- **Token Embedding**: 50,257 × 768 = ~38.6M parameters
- **Positional Embedding**: 256 × 768 = ~0.2M parameters
- **12 Transformer Blocks**: Each with ~7.1M parameters = ~85.2M parameters
- **Total**: ~124 Million parameters

---

## Text Generation Mechanism

### Generation Strategy: Greedy Decoding

The model generates text token-by-token, selecting the most likely next token at each step:

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # Crop to context size
        idx_cond = idx[:, -context_size:]
        
        # Get model predictions
        with torch.no_grad():
            logits = model(idx_cond)  # (batch, seq_len, vocab_size)
        
        # Focus on last token
        logits = logits[:, -1, :]  # (batch, vocab_size)
        
        # Get probabilities
        probas = torch.softmax(logits, dim=-1)
        
        # Select token with highest probability
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

### How Generation Works

1. **Input Encoding**: Convert prompt text to token IDs
2. **Forward Pass**: Pass tokens through the model to get logits
3. **Probability Distribution**: Apply softmax to get probability distribution over vocabulary
4. **Token Selection**: Choose next token (greedy: highest probability, or sampling for diversity)
5. **Sequence Extension**: Append new token to sequence
6. **Repeat**: Continue until reaching max length or end-of-sequence token

### Example Generation Flow

```
Input: "Hello, I am"  →  [31373, 11, 314, 716]  (4 tokens)
Model processes → logits for position 4
Softmax → probabilities for all 50,257 vocab tokens
Argmax → token with highest probability: say 262 ("Super")
New sequence: [31373, 11, 314, 716, 262]
Repeat...
Output: "Hello, I am Super intelligent and curious..."
```

### Alternative Generation Methods

**Temperature Sampling:**
- Adjust softmax temperature to control randomness
- Temperature < 1.0: More deterministic (sharper distribution)
- Temperature > 1.0: More random (softer distribution)

**Top-K Sampling:**
- Only sample from top K most likely tokens
- Prevents very unlikely tokens

**Top-P (Nucleus) Sampling:**
- Sample from smallest set of tokens with cumulative probability > P
- Balances diversity and quality

---

## Training Costs and Computational Complexity

### Why Training Transformers is Expensive

#### 1. **Computational Complexity**

**Attention Mechanism Complexity:**
- Self-attention: $O(n^2 \cdot d)$ where $n$ = sequence length, $d$ = embedding dimension
- For GPT-2: $O(256^2 \cdot 768) = O(50,331,648)$ operations per sample
- With 12 heads and 12 layers: Multiplied by ~144

**Matrix Multiplications:**
- Forward pass requires numerous large matrix multiplications
- Backward pass (gradients) requires additional computations
- GPU memory grows quadratically with sequence length

#### 2. **Memory Requirements**

**Activation Storage:**
- All activations must be kept for backpropagation
- For batch size 32, seq_len 256, emb_dim 768, 12 layers:
  - Memory ≈ 32 × 256 × 768 × 12 × 4 bytes ≈ 37.6 GB
  - Plus gradients and optimizer states (another 2-3x multiplier)

**Quadratic Memory with Sequence Length:**
- Attention mechanism creates $n \times n$ matrices
- Doubling sequence length quadruples memory needs

#### 3. **Training Time**

**Per-Sample Processing:**
- Each training sample requires:
  - Forward pass through 12 layers × 12 heads
  - Backward pass through same path
  - Parameter updates

**Typical Training Schedule:**
- Large language models train for weeks/months on thousands of GPUs
- GPT-2 (124M): ~1 week on 8 GPUs
- Larger models like GPT-3 (175B): Months on thousands of GPUs

#### 4. **Data Preprocessing**

**Tokenization Overhead:**
- Converting billions of tokens requires significant preprocessing
- Storing and accessing training data is a bottleneck

**Creating Datasets:**
- Sliding window approach creates overlapping samples
- For 256-token context with stride 128: 2x more data points
- Billions of tokens create millions of training samples

#### 5. **Hardware Requirements**

**GPU Memory:**
- Single GPU (24GB): Supports batch size ~2-4 for 124M model
- Multi-GPU training requires distributed training implementation
- All-reduce communications add overhead

**Data Loading:**
- CPUs must continuously feed data to GPUs
- I/O becomes bottleneck with insufficient bandwidth

### Cost Breakdown Example

Training GPT-2 124M for one epoch on "The Verdict" (~300K tokens):

```
Tokens in dataset: 300,000
Context length: 256
Stride: 128
Samples created: ~2,300

Forward pass per sample:
  - 12 layers × 12 heads × O(256^2 × 64) ≈ 50M ops
  - Total: ~2.3 × 50M = 115B operations

Backward pass: Same as forward (backward is similar computational cost)

Per-epoch cost: ~230B operations

With modern GPUs (100-300 TFLOPS): ~1-2 seconds per epoch
With multiple epochs (10-20): 10-40 seconds total

Larger models:
- GPT-2 1.5B parameters: 12-24x more computation
- Training time scales super-linearly with model size
```

### The Training Loop: Step-by-Step

Understanding what happens during training reveals why costs are so high:

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device, 
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, tokens_seen = [], [], []
    tokens_seen_count, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()  # Enable dropout, batch norm updates
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Clear old gradients
            
            # 1. FORWARD PASS: Compute predictions
            logits = model(input_batch)  # (batch_size, seq_len, vocab_size)
            
            # 2. LOSS COMPUTATION: Compare with targets
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),      # Flatten to (batch*seq_len, vocab_size)
                target_batch.flatten()      # Flatten to (batch*seq_len,)
            )
            
            # 3. BACKWARD PASS: Compute gradients
            loss.backward()  # ← This is the expensive part!
            
            # 4. OPTIMIZER STEP: Update parameters
            optimizer.step()
            
            # Track metrics
            tokens_seen_count += input_batch.numel()
            global_step += 1
            train_losses.append(loss.item())
            
            # Validation
            if global_step % eval_freq == 0:
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                val_losses.append(val_loss)
```

#### Phase 1: Forward Pass - $O(n^2)$ Complexity

**What Happens:**
- Input tokens pass through embedding layers
- Each of 12 transformer blocks processes the sequence
- Each block performs 12-head attention where each head computes $n \times n$ attention matrix

**Concrete Numbers for Single Sample:**
```
Batch size: 1
Sequence length (n): 256
Embedding dim (d): 768
Number of heads: 12
Head dimension: 64

Per attention head:
  Q, K, V projections: 3 × (256 × 768 × 768) = ~450M operations
  Attention scores: (256 × 256 × 64) = ~4M operations
  Softmax: 256 × 256 = ~65K operations
  Context: (256 × 256) × 64 = ~4M operations

Single head cost: ~454M operations
12 heads × 12 layers = 144 heads total
Per-sample forward: ~65B operations
```

**Memory Usage During Forward Pass:**
- Activations must be saved for gradient computation
- Memory grows linearly with layers but quadratically with sequence length
- Batch size 8 with 256 tokens: ~4-8 GB GPU memory just for activations

#### Phase 2: Loss Computation

**Cross-Entropy Loss Function:**

$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \log(\text{softmax}(\text{logits}_i)[\text{target}_i])$$

Where:
- $N$ = batch_size × seq_len = 8 × 256 = 2,048 positions
- Each position has 50,257 possible tokens
- Softmax computes: $e^x / \sum e^x$ for all 50,257 values

**Cost per Batch:**
```
Softmax over 50,257 tokens at 2,048 positions:
  exp(): 2,048 × 50,257 = ~103M operations
  sum(): 2,048 × 50,257 = ~103M operations
  division: 2,048 × 50,257 = ~103M operations
  log: 2,048 = negligible

Loss computation: ~309M operations per batch
```

#### Phase 3: Backward Pass - The Expensive Part

**Why Backward Pass is Expensive:**

The backward pass must:
1. Compute gradients for every parameter
2. Apply chain rule through all 12 layers
3. Handle attention gradients (most expensive)

**Gradient Flow Through Attention:**

$$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial \text{attn\_out}} \cdot \frac{\partial \text{attn\_out}}{\partial Q}$$

Each component requires backpropagating through:
- Softmax: $\frac{\partial L}{\partial \text{scores}} = \text{attn\_weights} \odot (\text{grad} - \text{weighted\_grad})$
- Matrix multiply: Requires transpose and another matrix multiply
- Linear projections: Gradients for $W_Q, W_K, W_V$

**Backward Pass Cost:**

The backward pass typically requires **2-3x the forward pass cost** due to:
- Storing intermediate activations
- Computing gradients at each node
- Chain rule applications through dense layers

```
Forward pass: ~65B operations per sample
Backward pass: ~130-195B operations per sample
Total per sample: ~195-260B operations
```

#### Phase 4: Parameter Updates

**Optimizer Step (Adam Example):**

For each of 124M parameters:

```python
# Standard SGD update
param = param - learning_rate * gradient

# Adam update (more common, more expensive)
m = beta1 * m + (1 - beta1) * grad           # First moment
v = beta2 * v + (1 - beta2) * grad^2         # Second moment
m_hat = m / (1 - beta1^t)                    # Bias correction
v_hat = v / (1 - beta2^t)                    # Bias correction
param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

**Memory Overhead:**
- Standard gradients: 124M × 4 bytes = 496 MB
- Adam maintains two additional buffers (m, v): +992 MB
- Total: ~1.5 GB just for optimizer state

This is why Adam training requires 3x more GPU memory than SGD!

### Why Training Costs Explode

#### Exponential Scaling with Model Size

**Parameter Count:**
```
GPT-2 models:
  - Small (125M): ~125M parameters
  - Medium (355M): ~355M parameters  
  - Large (774M): ~774M parameters
  - XL (1.5B): ~1.5B parameters

Scaling relationship:
  Cost ∝ (# parameters) × (# tokens)
  
  1.5B vs 125M: 12× parameters = ~12× training time
```

**Compute Requirements (FLOPs):**

For one training iteration:

$$\text{FLOPs} = 6 \times (\text{parameters} \times \text{seq\_len} \times \text{batch\_size} \times \text{num\_epochs})$$

The factor of 6 comes from:
- 2× for forward and backward pass
- 3× for attention quadratic complexity

**Real-World Examples:**

```
GPT-2 (125M parameters):
  Training data: 40 GB (Wikipedia + Books)
  Tokens: ~8 billion
  Batch size: 512, Seq_len: 1024
  
  Estimated FLOPs: 
    6 × 125M × 1024 × 512 × 3 epochs ≈ 1.2 × 10^18 FLOPs
  
  Time on RTX 3090 (142 TFLOPS):
    1.2×10^18 / 142×10^12 ≈ 8,450 GPU-hours ≈ 1-2 weeks

GPT-3 (175B parameters):
  Training data: 300 GB (CommonCrawl + Books)
  Tokens: ~300 billion
  Batch size: 3,200, Seq_len: 2048
  
  Estimated FLOPs:
    6 × 175B × 2048 × 3200 × 1 epoch ≈ 6.8 × 10^21 FLOPs
  
  Time on 1,024 A100 GPUs (312 TFLOPS each):
    6.8×10^21 / (1024 × 312×10^12) ≈ 21,000 GPU-days ≈ 3 months
  
  Estimated cost: $5-10 million USD
```

#### Sequence Length Quadratic Effect

**Attention Complexity Comparison:**

```
Sequence length effect on attention:
  256 tokens: 65,536 attention operations per head
  512 tokens: 262,144 attention operations per head  (4× more)
  1024 tokens: 1,048,576 attention operations per head (16× more)
  2048 tokens: 4,194,304 attention operations per head (64× more)

Total effect on training:
  Doubling sequence length:
    - 4× more attention compute
    - 4× more activations to store (memory)
    - Forward pass ~2× slower
    - Backward pass ~2× slower
    - Can quadruple training time
```

#### Batch Size Trade-offs

Larger batches enable better GPU utilization but hit memory limits:

```
GPT-2 on single 40GB GPU:
  Batch size 1:   uses ~12GB GPU memory
  Batch size 8:   uses ~30GB GPU memory
  Batch size 16:  uses ~38GB GPU memory (max)
  Batch size 32:  requires GPU memory we don't have!

Solution: Distributed training
  32 GPUs with batch 1 each = effective batch 32
  Communication overhead between GPUs: ~10-20% extra time
```

#### Validation During Training

**Why Validation is Expensive:**

```python
# Validation loop (no backward, but still expensive)
for input_batch, target_batch in val_loader:
    with torch.no_grad():  # No gradient computation
        logits = model(input_batch)
    loss = cross_entropy(logits, target_batch)
```

**Cost per epoch:**
- Forward pass through entire validation set
- Can be 10-20% of training set size
- Runs every N steps (e.g., every 500 steps)
- Adds ~5-10% overhead to total training time

```
Example: Every 500 training steps, validate on 10K validation samples
  Each validation: 10K/8 batch_size = 1,250 forward passes
  Cost: ~81B FLOPs
  
  If training for 100K steps with eval every 500 steps:
    200 validation runs × 81B = 16.2T FLOPs extra
    Total training: 1.2×10^18 FLOPs + 16.2T ≈ +0.001% overhead
```

But validation prevents overfitting, so essential despite the cost!

### Optimization Strategies to Reduce Costs

**Gradient Checkpointing:**
- Trade computation for memory
- Recompute activations during backward pass instead of storing them
- Implementation:
  ```python
  # Without checkpointing
  x = transformer_block(x)  # Saves all activations
  
  # With checkpointing
  x = torch.utils.checkpoint.checkpoint(transformer_block, x)
  # Only saves inputs, recomputes during backward
  ```
- **Result**: 50% memory reduction, 30% time increase
- **When to use**: When memory is bottleneck, not speed

**Mixed Precision Training (float16):**
- Typical training: float32 (32-bit, ~7 decimal places precision)
- Mixed precision: float16 (16-bit) for forward/backward, float32 for updates
  ```
  Forward pass:  float16  (faster on GPUs, less memory)
  Backward pass: float16  (less memory)
  Parameter update: float32 (maintains precision)
  ```
- **Result**: 2× speedup, 2× less memory, minimal accuracy loss (<0.1%)

**Learning Rate Scheduling:**
- Start with high LR, gradually decrease
  ```python
  # Cosine annealing
  lr = 0.5 * base_lr * (1 + cos(π × step / total_steps))
  ```
- Prevents oscillations, improves final accuracy
- Can reduce required epochs by 20-30%

**Distributed Training:**

```
Single GPU: 1 GPU, 1 sample batch_size
Data parallelism: 8 GPUs, batch_size 1 per GPU = effective batch 8
  - Computation: ~8× faster (ideal scaling)
  - Communication: ~10% overhead (gradients allreduce)
  - Effective: ~7× faster

Multi-node: 64 GPUs across 8 nodes
  - Same benefits as above
  - Network communication: ~20-30% overhead
  - Effective: ~5-6× faster on 64 GPUs
```

**FlashAttention (Algorithm Level Optimization):**
- Fuses attention operations into single CUDA kernel
- Reduces memory access (I/O bound vs compute bound)
- Implementation details: IO-aware attention algorithm
- **Result**: 3-5× faster attention, similar memory usage

**Sparse Attention Patterns:**
- Only attend to nearby tokens instead of all tokens
- Reduces attention from $O(n^2)$ to $O(n \log n)$ or $O(n)$
- Variants: local attention, strided attention, bigbird patterns
- **Trade-off**: Slightly worse model quality for significant speedup

**Quantization (Post-Training):**
- Reduce weights from float32 to int8 or lower
- Applied after training completes
- Doesn't speed up training but enables deployment on smaller hardware
- **Trade-off**: Slight accuracy loss (~0.5-2%)

### Real-World Cost Comparison

**Training on Different Hardware:**

```
Training GPT-2 (125M) for 1 epoch on 8B tokens:

Single RTX 3090 (142 TFLOPS):
  Estimated time: ~100 hours = 4.2 days
  Cost: ~$5 (assuming $0.05/hour electricity)

8× RTX 3090 with data parallelism:
  Estimated time: ~12.5 hours
  Cost: ~$5 (same per-GPU cost, 8 GPUs)

Single A100 (312 TFLOPS):
  Estimated time: ~46 hours = 1.9 days
  Cost: ~$50 (assuming $50/hour cloud GPU)

8× A100:
  Estimated time: ~6 hours
  Cost: ~$400

GPT-3 on 1,024 × A100:
  Estimated time: ~3 months
  Cost: $5-10 million
```

---

## Key Learnings

### 1. **Attention is Powerful but Expensive**

- Attention mechanism is the core innovation enabling transformers
- Query-Key-Value framework allows dynamic feature selection
- Causal masking is essential for autoregressive language modeling
- However, $O(n^2)$ complexity limits scalability

### 2. **Residual Connections Enable Deep Networks**

- Without residuals, gradients cannot flow through 12+ layers
- Residuals preserve information and improve optimization
- Modern architectures build on this foundation

### 3. **Positional Information is Crucial**

- Transformers are permutation-invariant without positional embeddings
- Positional encoding allows understanding of sequence order
- Affects model's ability to learn sequential relationships

### 4. **The Transformer Block is Modular**

Each block repeats:
- Normalize → Attention → Residual
- Normalize → FFN → Residual

This modularity enables:
- Stacking many layers
- Transfer learning
- Fine-tuning on downstream tasks

### 5. **Training Costs Scale Dramatically**

- Compute scales with model size (parameters) and data size
- Memory requirements scale quadratically with sequence length
- Hardware constraints often determine practical model size
- Efficiency optimizations are critical for practical training

### 6. **Generation Requires Sampling Strategies**

- Greedy decoding is deterministic but repetitive
- Temperature/top-K sampling add diversity
- Trade-off between quality and diversity
- Different applications need different strategies

### 7. **Dataset Preparation is Critical**

- Tokenization choices affect model capacity
- Train/validation split prevents overfitting
- Sliding window creates overlapping samples
- Data quality dominates model quality (garbage in, garbage out)

---

## Conclusion

Building GPT-2 from scratch reveals the elegant architecture underlying modern language models. The transformer's power comes from:

1. **Self-attention** mechanism for dynamic relationship modeling
2. **Multi-head** architecture for diverse feature learning
3. **Residual connections** enabling very deep networks
4. **Efficient parallel computation** on modern hardware

However, the trade-offs are significant:

- Quadratic memory/compute in sequence length limits context size
- Training costs restrict practical applications to well-funded organizations
- Inference remains computationally demanding for real-time applications

Understanding these components provides insight into both capabilities and limitations of modern large language models. The fundamental principles learned here apply to GPT-3, GPT-4, and other contemporary architectures.

---

**Author's Note:** This documentation synthesizes concepts from Sebastian Raschka's "Building a Large Language Model from Scratch" with practical implementation details from the GPT-2 scratch project.
