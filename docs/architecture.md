# MoPE-Transformer Architecture

This document summarizes how the repository maps the Mixture of Pipeline Experts idea onto a minimal, hackable Python scaffold.

## MoPE layer design
Each MoPE layer replaces the FFN portion of a Transformer block with three parts:

1. **Gate:** projects a hidden state to a categorical distribution over pipeline experts.
2. **Pipeline executor:** runs the selected pipeline (a sequence of atomic reasoning tools such as `planner`, `search`, `reader`, `verifier`).
3. **Vectorizer:** converts the textual pipeline output back into a vector in the same dimensionality as the hidden state and applies a residual connection.

```text
[hidden state] --(Gate)--> pipeline id --(execute pipeline)--> textual trace --(Vectorizer)--> vector update --(residual)--> next hidden state
```

The current code uses pure-Python numerics (no external ML dependencies) to keep the prototype self-contained; swapping in PyTorch or JAX is straightforward because the interfaces are cleanly separated.

## Pipeline experts
Pipeline experts are defined in `mope/pipeline.py` as ordered lists of atomic steps. Example experts provided:

- **Planner → Search → Reader → Verifier**: careful, higher-cost strategy that explains what to search, gathers snippets, and double-checks them.
- **Search → Reader**: cheaper strategy for straightforward fact lookup.
- **Planner → Search**: planner-guided single-pass search.
- **MultiSearch → Compare → Verify**: multi-source comparison before verifying consensus.

Each expert returns both an answer string and a detailed trace of the intermediate reasoning steps for transparency.

## Gate strategies
`mope/gate.py` contains a simple linear gate that produces logits over experts and samples the argmax route. The gate keeps full probability distributions so training objectives (e.g., distillation, RL) can operate on soft decisions.

In research settings, the gate can be swapped for:
- Contextual bandits that account for search cost
- Multi-step policies that revisit gating mid-layer
- Temperature / entropy regularization to encourage exploration

## Vectorization
`mope/vectorizer.py` maps pipeline textual outputs into vector space using a deterministic bag-of-words hashing scheme. This keeps the example free of external embeddings while making the transformation stable enough to use in tests.

## Training curriculum
`mope/training.py` sketches three phases:
1. **FFN distillation:** fit MoPE outputs to a frozen FFN for stability (mean-squared error loss).
2. **Supervised fine-tuning (SFT):** teach pipelines to solve search/QA tasks via cross-entropy over answers and routing.
3. **Reinforcement learning (RL):** optimize a reward that balances accuracy, search cost, and factuality; adjust the gate accordingly.

The functions include structured placeholders, logging hooks, and TODOs to help researchers plug in their preferred frameworks.
