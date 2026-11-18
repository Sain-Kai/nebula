"""
Attention pattern extraction, head specialization analysis, and information flow tracing.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np


class AttentionAnalyzer:
    """
    Analyzes multi-head attention weights, measuring head entropy,
    identifying attention specialization, and generating token interaction matrices.
    """

    def __init__(self, n_heads: int):
        self.n_heads = n_heads

    def analyze_layer_attention(
        self,
        attn_weights: np.ndarray,
        token_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes attention weights of shape (batch, n_heads, seq_len, total_kv_len).
        """
        # Take first item in batch
        # Shape: (n_heads, seq_len, total_kv_len)
        matrix = attn_weights[0]
        n_heads, seq_len, kv_len = matrix.shape

        if token_labels is None:
            token_labels = [f"T{i}" for i in range(kv_len)]

        head_analyses: List[Dict[str, Any]] = []

        for h in range(n_heads):
            head_mat = matrix[h]  # (seq_len, kv_len)

            # Entropy across key dimension for each query token
            safe_weights = np.clip(head_mat, 1e-10, 1.0)
            entropy_per_query = -np.sum(head_mat * np.log(safe_weights), axis=-1)
            mean_entropy = float(np.mean(entropy_per_query))

            # Positional bias: how much weight falls on recent token vs first token
            first_token_attention = float(np.mean(head_mat[:, 0])) if kv_len > 0 else 0.0
            
            # Diagonal attention (self or previous token)
            diag_attention = 0.0
            if seq_len > 1 and kv_len >= seq_len:
                diag = np.diagonal(head_mat[:, -seq_len:])
                diag_attention = float(np.mean(diag))

            # Classification of head behavior
            if mean_entropy < 0.5:
                behavior = "Highly Focused"
            elif first_token_attention > 0.4:
                behavior = "BOS / Prefix Anchor"
            elif diag_attention > 0.4:
                behavior = "Previous/Self Token Tracker"
            else:
                behavior = "Broad Context Aggregator"

            head_analyses.append({
                "head_idx": h,
                "mean_entropy": round(mean_entropy, 4),
                "bos_attention": round(first_token_attention, 4),
                "diagonal_attention": round(diag_attention, 4),
                "classification": behavior,
                "matrix_sample": head_mat[-1].tolist() if seq_len > 0 else [],  # Last query attention
            })

        return {
            "seq_len": seq_len,
            "kv_len": kv_len,
            "head_analyses": head_analyses,
            "token_labels": token_labels[:kv_len],
        }
