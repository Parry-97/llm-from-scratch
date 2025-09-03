# 🤖 Building an LLM from Scratch

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red)
![tiktoken](https://img.shields.io/badge/tiktoken-0.11.0-green)
![uv](https://img.shields.io/badge/uv-package%20manager-7f52ff)
![Progress](https://img.shields.io/badge/Book_Progress-Chapter_4%2F7-orange)

> A step-by-step implementation of a GPT-like Large Language Model following Sebastian Raschka's "Build a Large Language Model (From Scratch)"

## 📖 About This Project

This repository documents my journey through **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka. I'm implementing each concept from the book in PyTorch, building a GPT-like language model from the ground up to truly understand how modern LLMs work.

### 📚 Book Progress: Chapter 4 of 7

Currently implementing: **"Implementing a GPT model from scratch to generate text"**

#### ✅ Completed Chapters:
- **Chapter 1**: Understanding large language models
- **Chapter 2**: Working with text data 
- **Chapter 3**: Coding attention mechanisms
- **Chapter 4**: Implementing a GPT model from scratch to generate text *(in progress)*

#### 🔜 Upcoming Chapters:
- **Chapter 5**: Pretraining on unlabeled data
- **Chapter 6**: Fine-tuning for classification
- **Chapter 7**: Fine-tuning to follow instructions

## 🎯 Learning Objectives

By following along with the book and this implementation, I'm learning:

- **Fundamentals of LLMs**: How transformers revolutionized NLP and the architecture behind GPT models
- **Text Processing**: Tokenization strategies, vocabulary building, and data preparation for neural networks
- **Attention Mechanisms**: The mathematics and intuition behind self-attention and multi-head attention
- **Model Architecture**: How to build a complete GPT model with embeddings, transformer blocks, and generation capabilities
- **Training Strategies**: Pretraining objectives, loss functions, and optimization techniques (upcoming)
- **Fine-tuning**: Adapting pretrained models for specific tasks (upcoming)

## 🏗️ Current Implementation Status

### ✨ Implemented Components

#### 📝 **Text Data Processing (Chapter 2)**
- **SimpleTokenizerV1** ([tokenizer/simple_tokenizer.py](tokenizer/simple_tokenizer.py)): Custom regex-based tokenizer
- Text splitting and preprocessing utilities
- Vocabulary management and encoding/decoding
- Dataset preparation for training

#### 🎯 **Attention Mechanisms (Chapter 3)**
- **Simple Attention**: Basic attention implementation for understanding
- **Causal Attention**: Masked attention to prevent looking ahead
- **Multi-Head Attention** ([attention/multi_head_attention.py](attention/multi_head_attention.py)): Parallel attention heads with projection
- Scaled dot-product attention with proper normalization

#### 🤖 **GPT Model Architecture (Chapter 4 - Current Focus)**
- **DummyGPTModel** ([gpt_architecture/dummy_gpt_model.py](gpt_architecture/dummy_gpt_model.py)): Complete GPT model implementation
- **TransformerBlock** ([gpt_architecture/transformer.py](gpt_architecture/transformer.py)): Core transformer building block
- **FeedForward Networks**: Position-wise feed-forward with GELU activation
- **Layer Normalization**: Custom implementation for training stability
- Positional embeddings (learned)
- Residual connections and dropout

#### 🔤 **Text Generation (Chapter 4 - In Progress)**
- Token sampling strategies
- Temperature-based generation
- Basic greedy decoding

## 📁 Project Structure

```
llm-from-scratch/
├── attention/                    # Chapter 3: Attention implementations
│   ├── multi_head_attention.py  # Multi-head attention mechanism
│   ├── self_attention.py        # Self-attention basics
│   ├── causal_attention.py      # Masked attention for autoregression
│   ├── simple_attention.py      # Simplified attention for learning
│   └── *.md                     # Documentation and notes
├── gpt_architecture/             # Chapter 4: GPT model components
│   ├── dummy_gpt_model.py       # Main GPT model class
│   ├── transformer.py           # Transformer block
│   ├── feed_forward.py          # FFN layer
│   ├── layer_normalization.py   # LayerNorm implementation
│   └── gelu.py                  # GELU activation
├── tokenizer/                    # Chapter 2: Text processing
│   ├── simple_tokenizer.py      # Tokenizer implementation
│   ├── gpt_dataset.py          # Dataset utilities
│   ├── sampling.py             # Generation sampling methods
│   ├── text_splitting.py       # Text preprocessing
│   └── the-verdict.txt         # Sample text data
├── tests/                       # Unit tests
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Git
- (Optional) CUDA-capable GPU for faster computation

### Setup Instructions

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd llm-from-scratch
```

2. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Create and activate virtual environment:**
```bash
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. **Install dependencies:**
```bash
uv sync
```

### PyTorch GPU Support

The project uses PyTorch 2.4.0 (CPU version by default). For GPU support:

1. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/)
2. Select your configuration and install:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

## 💻 Usage Examples

### Basic Tokenization (Chapter 2)

```python
from tokenizer.simple_tokenizer import SimpleTokenizerV1

# Create tokenizer with vocabulary
vocab = {
    "Hello": 0, ",": 1, " ": 2, "world": 3, "!": 4,
    "LLM": 5, "from": 6, "scratch": 7
}
tokenizer = SimpleTokenizerV1(vocab)

# Encode and decode text
text = "Hello, world!"
token_ids = tokenizer.encode(text)
print(f"Tokens: {token_ids}")
print(f"Decoded: {tokenizer.decode(token_ids)}")
```

### Attention Mechanism (Chapter 3)

```python
import torch
from attention.multi_head_attention import MultiHeadAttention

# Setup multi-head attention
batch_size, seq_len, d_model = 2, 10, 768
mha = MultiHeadAttention(
    d_in=d_model,
    d_out=d_model, 
    context_length=seq_len,
    dropout=0.1,
    num_heads=12
)

# Process input
x = torch.randn(batch_size, seq_len, d_model)
output = mha(x)
print(f"Output shape: {output.shape}")  # [2, 10, 768]
```

### GPT Model Forward Pass (Chapter 4)

```python
import torch
from gpt_architecture.dummy_gpt_model import DummyGPTModel

# Model configuration
config = {
    "vocab_size": 5000,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Initialize model
model = DummyGPTModel(config)

# Forward pass
input_ids = torch.randint(0, config["vocab_size"], (2, 10))
with torch.no_grad():
    logits = model(input_ids)
print(f"Logits shape: {logits.shape}")  # [2, 10, 5000]
```

## 🔬 Technical Implementation Details

### Current Architecture (Chapter 4)

```
Input Text
    ↓
[Tokenization]
    ↓
Token IDs → Token Embeddings + Positional Embeddings
    ↓
[Transformer Block] × N_LAYERS
    ├── Multi-Head Attention (with causal mask)
    ├── Add & Norm
    ├── Feed-Forward Network
    └── Add & Norm
    ↓
[Final Layer Norm]
    ↓
[Output Projection] → Logits
    ↓
[Sampling/Generation] → Generated Text
```

### Key Design Decisions

- **Tokenizer**: Simple regex-based splitting (will explore BPE in later chapters)
- **Attention**: Scaled dot-product with causal masking for autoregression
- **Positional Encoding**: Learned embeddings (not sinusoidal)
- **Activation**: GELU in feed-forward networks
- **Normalization**: Pre-norm architecture (LayerNorm before sub-layers)
- **Model Size**: Configurable, default similar to GPT-2 small (768 dim, 12 heads, 12 layers)

## 🛠️ Technologies Used

- **[Python](https://python.org)** 3.11+: Core language
- **[PyTorch](https://pytorch.org)** 2.4.0: Deep learning framework
- **[NumPy](https://numpy.org)** 2.3.2+: Numerical operations
- **[tiktoken](https://github.com/openai/tiktoken)** 0.11.0+: OpenAI's BPE tokenizer (for comparison)
- **[uv](https://github.com/astral-sh/uv)**: Fast Python package management

### Development Tools
- **pytest**: Testing framework
- **IPython**: Interactive development
- **matplotlib**: Visualizations

## 📚 References & Resources

### Primary Reference
> **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka
> - [Book on Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch)
> - [Official GitHub Repository](https://github.com/rasbt/LLMs-from-scratch)
> - [Author's Website](https://sebastianraschka.com/)

### Additional Resources
- [Attention Is All You Need (Original Transformer Paper)](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

## 🚧 Roadmap

### Immediate Next Steps (Chapter 4 completion)
- [ ] Implement temperature-based sampling
- [ ] Add top-k and top-p (nucleus) sampling
- [ ] Create text generation utilities
- [ ] Build interactive generation demo

### Upcoming Chapters
- [ ] **Chapter 5**: Implement pretraining loop
- [ ] **Chapter 5**: Add data loading pipeline
- [ ] **Chapter 5**: Implement training metrics and logging
- [ ] **Chapter 6**: Add classification head
- [ ] **Chapter 6**: Implement fine-tuning procedures
- [ ] **Chapter 7**: Instruction following capabilities
- [ ] **Chapter 7**: RLHF concepts

### Future Enhancements
- [ ] Add comprehensive test coverage
- [ ] Create Jupyter notebooks for each chapter
- [ ] Build web interface with Gradio
- [ ] Add model checkpointing
- [ ] Performance profiling and optimization
- [ ] Docker containerization

## 🤝 Contributing

This is a personal learning project following the book's progression. However, I welcome:
- Bug reports and fixes
- Clarifications and documentation improvements  
- Discussions about the concepts
- Suggestions for better implementations

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- **Sebastian Raschka** for writing this excellent book and making LLMs accessible
- The PyTorch team for the amazing framework
- The open-source community for inspiration and resources

---

<div align="center">
<i>"The best way to understand something is to build it from scratch"</i><br>
🧠 Currently learning at Chapter 4/7 of the book 📚
</div>

