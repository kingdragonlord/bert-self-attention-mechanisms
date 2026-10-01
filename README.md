# Demystifying Self-Attention in BERT

A hands-on, step-by-step implementation of bidirectional self-attention mechanisms using PyTorch and Hugging Face Transformers. 

This notebook breaks down the internal mechanics of Transformer attention layers by walking through tokenization, embedding extraction with `bert-base-uncased`, linear projections into Query ($Q$), Key ($K$), and Value ($V$) spaces, and calculating context vectors for individual tokens without causal masking.

---

## 📌 Features

- **Tokenization & Embeddings**: Tokenizes input sequences and extracts 768-dimensional contextual token embeddings using `bert-base-uncased`.
- **Query, Key, & Value Projections**: Demonstrates how custom linear transformation layers map hidden dimensions ($d=768$) to projection spaces ($d_k = d_v = 64$).
- **Raw Attention Scores**: Computes dot-product compatibility scores across sequence positions:
  $$\text{Score}(q, K) = q K^T$$
- **Softmax Normalization**: Applies softmax across sequence dimension to obtain valid probability distributions over input tokens.
- **Attention Visualization**: Generates stem plots using Matplotlib comparing raw dot-product scores against normalized softmax weights.
- **Context Vector Computation**: Aggregates value vectors weighted by attention probabilities to construct enriched context representations ($z_i$).
- **Multi-Head Dimension Math**: Verifies projection shapes and calculates head distribution requirements ($h = d / d_v$).

---

## 🛠️ Tech Stack & Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/) (`torch`, `torch.nn`)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (`transformers`)
- [Matplotlib](https://matplotlib.org/)

Install dependencies via pip:

```bash
pip install torch transformers matplotlib
