"""
Mixture of Experts (MoE) implementation with gating teardown and routing diagnostics.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from nebula.core.layers import FeedForward


class MoERouter:
    """
    Learned router network for Mixture of Experts.
    Computes gating logits, selects top-k experts per token, and calculates routing entropy.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        use_bias: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.use_bias = use_bias

        if rng is None:
            rng = np.random.RandomState(42)

        scale = 1.0 / math.sqrt(d_model)
        self.w_gate = rng.randn(d_model, num_experts).astype(np.float32) * scale
        if use_bias:
            self.b_gate = np.zeros(num_experts, dtype=np.float32)

    def route(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Calculates expert assignments and weights for input tensor x.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            top_k_indices: Shape (batch, seq_len, top_k)
            top_k_weights: Shape (batch, seq_len, top_k)
            router_diagnostics: Dict with full probability distributions and entropy
        """
        logits = np.matmul(x, self.w_gate)
        if self.use_bias:
            logits += self.b_gate

        # Softmax over expert dimension
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

        # Select Top-K experts
        top_k_indices = np.argsort(-probs, axis=-1)[..., : self.top_k]
        top_k_weights_raw = np.take_along_axis(probs, top_k_indices, axis=-1)

        # Renormalize top-k gating weights so sum = 1.0
        weight_sum = np.sum(top_k_weights_raw, axis=-1, keepdims=True) + 1e-12
        top_k_weights = top_k_weights_raw / weight_sum

        # Router entropy: measure of routing uncertainty (higher = less specialized)
        safe_probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(safe_probs), axis=-1)

        # Expert utilization histogram across batch and seq_len
        flat_chosen = top_k_indices.reshape(-1)
        expert_counts = np.bincount(flat_chosen, minlength=self.num_experts)

        diagnostics = {
            "router_logits": logits,
            "all_expert_probs": probs,
            "top_k_indices": top_k_indices,
            "top_k_weights": top_k_weights,
            "routing_entropy": entropy,
            "expert_utilization": expert_counts.tolist(),
        }

        return top_k_indices, top_k_weights, diagnostics


class MoELayer:
    """
    Mixture of Experts layer composing multiple specialized FeedForward networks
    coordinated by an inspectable MoERouter.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        top_k: int = 2,
        activation_fn: str = "gelu",
        use_bias: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k

        if rng is None:
            rng = np.random.RandomState(42)

        self.router = MoERouter(d_model, num_experts, top_k=top_k, rng=rng)
        self.experts = [
            FeedForward(d_model, d_ff, activation_fn=activation_fn, use_bias=use_bias, rng=rng)
            for _ in range(num_experts)
        ]

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Dispatches tokens through selected experts and aggregates weighted outputs.
        """
        batch_size, seq_len, d_model = x.shape
        top_k_indices, top_k_weights, router_diag = self.router.route(x)

        # Output accumulator
        combined_output = np.zeros_like(x)
        expert_activations_summary: Dict[int, Any] = {}

        # Evaluate each expert
        for exp_idx, expert in enumerate(self.experts):
            exp_out, exp_insp = expert.forward(x)
            expert_activations_summary[exp_idx] = exp_insp

            # Find where this expert was selected in top_k
            for k_slot in range(self.top_k):
                mask = top_k_indices[..., k_slot] == exp_idx
                if np.any(mask):
                    weight = top_k_weights[..., k_slot : k_slot + 1]
                    combined_output += mask[..., None] * (exp_out * weight)

        inspection = {
            "router": router_diag,
            "top_k_indices": top_k_indices,
            "top_k_weights": top_k_weights,
            "expert_summaries": expert_activations_summary,
        }

        return combined_output, inspection
