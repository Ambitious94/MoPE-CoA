"""Training curriculum sketches for MoPE-Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

from .mope_layer import MoPELayer


@dataclass
class DistillationBatch:
    hidden_states: Sequence[Sequence[float]]
    ffn_outputs: Sequence[Sequence[float]]
    prompts: List[str]


@dataclass
class SFTBatch:
    prompts: List[str]
    labels: List[str]


@dataclass
class RLSample:
    prompt: str
    reward: float
    trajectory: dict


@dataclass
class RouteBatch:
    """Supervised routing batch.

    labels should be expert names present in layer.config.expert_names.
    hidden_states should match layer.config.hidden_size.
    """
    prompts: List[str]
    hidden_states: Sequence[Sequence[float]]
    labels: List[str]


def _mean_squared_error(preds: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> float:
    total = 0.0
    count = 0
    for p_vec, t_vec in zip(preds, targets):
        for p, t in zip(p_vec, t_vec):
            diff = p - t
            total += diff * diff
            count += 1
    return total / max(count, 1)


def distill_ffn(layer: MoPELayer, batch: DistillationBatch, loss_fn: Callable[[Sequence[Sequence[float]], Sequence[Sequence[float]]], float] | None = None) -> float:
    """Stage 1: fit MoPE outputs to a frozen FFN target."""

    predictions = []
    for hidden, prompt in zip(batch.hidden_states, batch.prompts):
        output = layer.forward(hidden, prompt)
        predictions.append(output["hidden_state"])
    metric = loss_fn or _mean_squared_error
    return float(metric(predictions, batch.ffn_outputs))


def supervised_finetune(model, batch: SFTBatch, loss_fn: Callable[[List[str], List[str]], float] | None = None) -> float:
    """Stage 2: teach pipelines to emit correct answers for labeled prompts."""

    outputs = [model.forward(prompt)["layer_traces"][-1]["trace"][-1] for prompt in batch.prompts]
    metric = loss_fn or (lambda preds, targets: sum(p != t for p, t in zip(preds, targets)) / max(len(preds), 1))
    return float(metric(outputs, batch.labels))


def reinforce_gate(model, samples: Iterable[RLSample], update_fn: Callable[[float, dict], None]) -> float:
    """Stage 3: optimize gate using reinforcement learning style updates.

    The API is framework-agnostic; `update_fn` can log gradients or update
    parameters directly depending on the caller's setup.
    """

    total_reward = 0.0
    for sample in samples:
        trajectory = model.forward(sample.prompt)
        total_reward += sample.reward
        update_fn(sample.reward, trajectory)
    return float(total_reward)


def train_gate_supervised(layer: MoPELayer, batch: RouteBatch, lr: float = 0.05) -> float:
    """One-step supervised update of the gate to prefer labeled experts.

    Implements a simple gradient step on the gate's linear weights using the
    softmax-cross-entropy gradient: dL/dlogits = probs - onehot(target).
    This operates directly on the prototype's Python lists (weight, bias).

    Returns the average cross-entropy loss over the batch.
    """

    # Build expert name -> index map
    name_to_idx = {name: i for i, name in enumerate(layer.config.expert_names)}

    def _cross_entropy(probs: List[float], target_idx: int) -> float:
        import math
        eps = 1e-12
        return -math.log(max(probs[target_idx], eps))

    total_loss = 0.0
    count = 0

    for hidden, prompt, label in zip(batch.hidden_states, batch.prompts, batch.labels):
        if label not in name_to_idx:
            # skip samples with unknown labels
            continue
        target_idx = name_to_idx[label]

        # Forward to get probabilities; use gate directly to avoid vectorizer effects
        probs = layer.gate.probs(list(hidden))
        loss = _cross_entropy(probs, target_idx)
        total_loss += loss
        count += 1

        # Gradient w.r.t logits: g_j = probs[j] - onehot[j]
        grad_logits = [p - (1.0 if j == target_idx else 0.0) for j, p in enumerate(probs)]

        # Update weights and bias: logits_j = sum_i hidden[i] * W[i][j] + b[j]
        for j, g in enumerate(grad_logits):
            # bias update
            layer.gate.bias[j] -= lr * g
            # weights update
            for i, h_i in enumerate(hidden):
                layer.gate.weight[i][j] -= lr * g * h_i

    return total_loss / max(count, 1)
