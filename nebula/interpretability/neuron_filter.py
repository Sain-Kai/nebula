"""
Neuron-level activation filtering, sparsity diagnostics, and decision filter teardown.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np


class NeuronFilter:
    """
    Inspects neuron activations across feed-forward and MoE blocks,
    calculates firing sparsity, and filters or prunes neuron activations.
    """

    def __init__(self, activation_threshold: float = 0.05):
        self.activation_threshold = activation_threshold

    def analyze_layer_neurons(
        self,
        activations: np.ndarray,
        layer_idx: int,
        is_moe: bool = False,
        expert_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes neuron activations of shape (batch, seq_len, d_ff).
        
        Args:
            activations: Raw post-activation tensor
            layer_idx: Layer index
            is_moe: Whether this is an MoE layer
            expert_idx: Specific expert index if applicable
        """
        # Shape: (seq_len, d_ff)
        act_2d = activations[0]
        seq_len, d_ff = act_2d.shape

        # Average activation magnitude per neuron
        mean_abs_act = np.mean(np.abs(act_2d), axis=0)  # (d_ff,)
        max_act = np.max(act_2d, axis=0)
        min_act = np.min(act_2d, axis=0)

        # Sparsity: fraction of neurons below threshold
        active_mask = mean_abs_act > self.activation_threshold
        num_active = int(np.sum(active_mask))
        sparsity_pct = round((1.0 - (num_active / d_ff)) * 100, 2)

        # Top firing neurons
        top_indices = np.argsort(-mean_abs_act)[:15]
        top_neurons = []
        for idx in top_indices:
            top_neurons.append({
                "neuron_idx": int(idx),
                "mean_abs_activation": round(float(mean_abs_act[idx]), 4),
                "max_activation": round(float(max_act[idx]), 4),
                "min_activation": round(float(min_act[idx]), 4),
            })

        # Distribution buckets for histogram visualization
        hist_counts, bin_edges = np.histogram(act_2d, bins=10)
        distribution = {
            "counts": hist_counts.tolist(),
            "bin_edges": [round(float(b), 3) for b in bin_edges.tolist()],
        }

        return {
            "layer_idx": layer_idx,
            "is_moe": is_moe,
            "expert_idx": expert_idx,
            "total_neurons": d_ff,
            "active_neurons": num_active,
            "sparsity_pct": sparsity_pct,
            "top_neurons": top_neurons,
            "activation_distribution": distribution,
        }

    def apply_neuron_filter(
        self,
        activations: np.ndarray,
        filter_mode: str = "keep_top_k",
        top_k: int = 32,
        threshold: float = 0.1,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies a selective filter to neuron activations to study representation changes.
        Modes: 'keep_top_k', 'threshold_clamp', 'prune_top'.
        """
        filtered = activations.copy()
        d_ff = activations.shape[-1]

        if filter_mode == "keep_top_k":
            # Mask all neurons except top_k
            ranks = np.argsort(-np.abs(filtered), axis=-1)
            keep_mask = ranks < top_k
            filtered = np.where(keep_mask, filtered, 0.0)
        elif filter_mode == "threshold_clamp":
            filtered = np.where(np.abs(filtered) >= threshold, filtered, 0.0)
        elif filter_mode == "prune_top":
            # Ablate the top most active neurons to test fragility
            ranks = np.argsort(-np.abs(filtered), axis=-1)
            prune_mask = ranks < top_k
            filtered = np.where(prune_mask, 0.0, filtered)

        suppressed_count = int(np.sum(filtered == 0.0))
        total_elements = filtered.size

        return filtered, {
            "mode": filter_mode,
            "suppressed_percentage": round((suppressed_count / total_elements) * 100, 2),
        }
