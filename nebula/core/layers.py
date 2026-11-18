"""
Transformer core architectural layers with full state inspection hooks.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (approximated formulation)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def swiglu(x_gate: np.ndarray, x_up: np.ndarray) -> np.ndarray:
    """Swish Gated Linear Unit."""
    swish = x_gate * (1.0 / (1.0 + np.exp(-np.clip(x_gate, -50.0, 50.0))))
    return swish * x_up


class LayerNorm:
    """Standard Layer Normalization with inspectable gamma and beta."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        # x: (..., d_model)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        stats = {
            "mean": mean,
            "variance": var,
            "normalized": x_norm,
        }
        return out, stats


class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.weight = np.ones(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        normed = x / rms
        out = normed * self.weight
        return out, {"rms": rms, "normalized": normed}


class MultiHeadAttention:
    """
    Multi-Head Attention supporting autoregressive KV caching and complete
    attention score/weight extraction across all heads.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 256,
        use_bias: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.use_bias = use_bias

        if rng is None:
            rng = np.random.RandomState(42)

        # Xavier/Glorot normal initialization
        scale = 1.0 / math.sqrt(d_model)
        self.w_q = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.w_k = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.w_v = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.w_o = rng.randn(d_model, d_model).astype(np.float32) * scale

        if use_bias:
            self.b_q = np.zeros(d_model, dtype=np.float32)
            self.b_k = np.zeros(d_model, dtype=np.float32)
            self.b_v = np.zeros(d_model, dtype=np.float32)
            self.b_o = np.zeros(d_model, dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,
        past_k: Optional[np.ndarray] = None,
        past_v: Optional[np.ndarray] = None,
        causal_mask: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Forward pass for Multi-Head Attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            past_k: Cached keys of shape (batch, n_heads, past_seq_len, d_head)
            past_v: Cached values of shape (batch, n_heads, past_seq_len, d_head)
            causal_mask: Whether to apply lower-triangular causal mask
            
        Returns:
            out: (batch, seq_len, d_model)
            current_k: (batch, n_heads, total_seq_len, d_head)
            current_v: (batch, n_heads, total_seq_len, d_head)
            inspection: Dict containing attention weights, query/key activations, etc.
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = np.matmul(x, self.w_q)
        k = np.matmul(x, self.w_k)
        v = np.matmul(x, self.w_v)

        if self.use_bias:
            q += self.b_q
            k += self.b_k
            v += self.b_v

        # Reshape to (batch, n_heads, seq_len, d_head)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # Update KV cache
        if past_k is not None and past_k.shape[2] > 0:
            k = np.concatenate([past_k, k], axis=2)
            v = np.concatenate([past_v, v], axis=2)

        total_kv_len = k.shape[2]

        # Scaled dot-product attention: (batch, n_heads, seq_len, total_kv_len)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head)

        # Causal masking
        if causal_mask:
            # past_len = total_kv_len - seq_len
            past_len = total_kv_len - seq_len
            mask = np.triu(np.ones((seq_len, total_kv_len), dtype=bool), k=past_len + 1)
            scores = np.where(mask[None, None, :, :], -1e9, scores)

        # Softmax over key dimension
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)

        # Weighted sum of values: (batch, n_heads, seq_len, d_head)
        attn_output = np.matmul(attn_weights, v)

        # Concatenate heads: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Output projection
        out = np.matmul(attn_output, self.w_o)
        if self.use_bias:
            out += self.b_o

        inspection = {
            "query": q,
            "key": k,
            "value": v,
            "raw_scores": scores,
            "attention_weights": attn_weights,
            "head_outputs": attn_output,
        }

        return out, k, v, inspection


class FeedForward:
    """
    Feed-Forward Network (MLP) with explicit neuron activation tracking.
    Enables layer-wise inspection of which neurons fire for given tokens.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation_fn: str = "gelu",
        use_bias: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_fn = activation_fn
        self.use_bias = use_bias

        if rng is None:
            rng = np.random.RandomState(42)

        scale1 = 1.0 / math.sqrt(d_model)
        scale2 = 1.0 / math.sqrt(d_ff)

        self.w_gate = rng.randn(d_model, d_ff).astype(np.float32) * scale1
        self.w_up = rng.randn(d_model, d_ff).astype(np.float32) * scale1
        self.w_down = rng.randn(d_ff, d_model).astype(np.float32) * scale2

        if use_bias:
            self.b_gate = np.zeros(d_ff, dtype=np.float32)
            self.b_up = np.zeros(d_ff, dtype=np.float32)
            self.b_down = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through MLP.
        Captures pre-activation, activated state, and down-projection.
        """
        pre_act = np.matmul(x, self.w_gate)
        if self.use_bias:
            pre_act += self.b_gate

        if self.activation_fn == "gelu":
            activated = gelu(pre_act)
        elif self.activation_fn == "relu":
            activated = np.maximum(0, pre_act)
        elif self.activation_fn == "swiglu":
            up_act = np.matmul(x, self.w_up)
            if self.use_bias:
                up_act += self.b_up
            activated = swiglu(pre_act, up_act)
        else:
            activated = pre_act

        out = np.matmul(activated, self.w_down)
        if self.use_bias:
            out += self.b_down

        # Neuron statistics
        neuron_magnitudes = np.abs(activated)
        sparsity = float(np.mean(neuron_magnitudes < 1e-4))

        inspection = {
            "pre_activation": pre_act,
            "neuron_activations": activated,
            "sparsity": sparsity,
            "top_neuron_indices": np.argsort(-neuron_magnitudes, axis=-1)[..., :10],
            "top_neuron_values": np.take_along_axis(
                activated, np.argsort(-neuron_magnitudes, axis=-1)[..., :10], axis=-1
            ),
        }

        return out, inspection
