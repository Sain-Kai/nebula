"""
Mechanistic interpretability module for KV cache lifecycle and memory dynamics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class CacheSnapshot:
    """Snapshot of KV cache state at an inference step."""
    step: int
    current_token_idx: int
    total_cached_tokens: int
    memory_bytes: int
    layers_k_norm: List[float]
    layers_v_norm: List[float]
    layer_summaries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "current_token_idx": self.current_token_idx,
            "total_cached_tokens": self.total_cached_tokens,
            "memory_bytes": self.memory_bytes,
            "memory_kb": round(self.memory_bytes / 1024, 2),
            "layers_k_norm": self.layers_k_norm,
            "layers_v_norm": self.layers_v_norm,
            "layer_summaries": self.layer_summaries,
        }


class KVCacheTracker:
    """
    Tracks and visualizes the filling, growth, and retention of the Key-Value Cache
    across prompt prefill and sequential autoregressive decoding steps.
    """

    def __init__(self, n_layers: int, n_heads: int, d_head: int, max_seq_len: int = 256):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.history: List[CacheSnapshot] = []

    def record_step(
        self,
        step: int,
        past_kv: List[Tuple[np.ndarray, np.ndarray]],
        current_token_id: Optional[int] = None,
    ) -> CacheSnapshot:
        """
        Extracts metrics from the current past_kv list.
        past_kv is a list of (k, v) tuples of shape (batch, n_heads, seq_len, d_head).
        """
        if not past_kv or past_kv[0][0] is None:
            snapshot = CacheSnapshot(
                step=step,
                current_token_idx=0,
                total_cached_tokens=0,
                memory_bytes=0,
                layers_k_norm=[],
                layers_v_norm=[],
            )
            self.history.append(snapshot)
            return snapshot

        total_tokens = past_kv[0][0].shape[2]
        dtype_bytes = past_kv[0][0].itemsize

        total_bytes = 0
        k_norms: List[float] = []
        v_norms: List[float] = []
        layer_summaries: List[Dict[str, Any]] = []

        for l_idx, (k, v) in enumerate(past_kv):
            # Shape: (batch, n_heads, seq_len, d_head)
            layer_bytes = (k.size + v.size) * dtype_bytes
            total_bytes += layer_bytes

            k_norm = float(np.linalg.norm(k))
            v_norm = float(np.linalg.norm(v))
            k_norms.append(round(k_norm, 4))
            v_norms.append(round(v_norm, 4))

            # Mean activation energy per head
            head_energy = np.mean(np.abs(k), axis=(0, 2, 3)).tolist()

            layer_summaries.append({
                "layer_idx": l_idx,
                "cached_tokens": total_tokens,
                "layer_bytes": layer_bytes,
                "k_norm": round(k_norm, 4),
                "v_norm": round(v_norm, 4),
                "head_energy": [round(e, 4) for e in head_energy],
                "cache_fill_percentage": round((total_tokens / self.max_seq_len) * 100, 2),
            })

        snapshot = CacheSnapshot(
            step=step,
            current_token_idx=total_tokens - 1,
            total_cached_tokens=total_tokens,
            memory_bytes=total_bytes,
            layers_k_norm=k_norms,
            layers_v_norm=v_norms,
            layer_summaries=layer_summaries,
        )
        self.history.append(snapshot)
        return snapshot

    def get_summary(self) -> Dict[str, Any]:
        """Provides overall summary of cache trajectory."""
        return {
            "num_steps": len(self.history),
            "current_cached_tokens": self.history[-1].total_cached_tokens if self.history else 0,
            "max_seq_len": self.max_seq_len,
            "total_memory_kb": round((self.history[-1].memory_bytes / 1024), 2) if self.history else 0,
            "snapshots": [s.to_dict() for s in self.history[-20:]],  # Return recent snapshots
        }
