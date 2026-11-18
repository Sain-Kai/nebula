"""
Counterfactual activation patching and neuron-level steering engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import numpy as np


@dataclass
class PatchIntervention:
    """Specification of an intervention at a specific layer, neuron, or stream."""
    layer_idx: int
    target: str  # 'residual_stream', 'post_attn', 'post_ffn', 'neuron'
    action: str  # 'inject', 'replace', 'zero_ablate', 'scale'
    neuron_idx: Optional[int] = None
    vector: Optional[np.ndarray] = None
    scalar_multiplier: float = 1.0


class ActivationPatcher:
    """
    Coordinates activation patching experiments to study causal mechanisms.
    Supports injecting steering vectors, zero-ablating specific neurons/heads,
    and comparing counterfactual model trajectories.
    """

    def __init__(self):
        self.interventions: List[PatchIntervention] = []

    def add_intervention(self, intervention: PatchIntervention):
        self.interventions.append(intervention)

    def clear(self):
        self.interventions.clear()

    def build_hook(self) -> Callable[[int, str, np.ndarray], np.ndarray]:
        """Creates a forward hook matching NebulaTransformer signature."""
        def hook(layer_idx: int, target: str, activation: np.ndarray) -> np.ndarray:
            for item in self.interventions:
                if item.layer_idx == layer_idx and item.target == target:
                    if item.action == "zero_ablate":
                        if item.neuron_idx is not None and target in ("post_ffn", "neuron"):
                            activation = activation.copy()
                            activation[..., item.neuron_idx] = 0.0
                        else:
                            activation = np.zeros_like(activation)
                    elif item.action == "scale":
                        activation = activation * item.scalar_multiplier
                    elif item.action == "inject" and item.vector is not None:
                        activation = activation + item.vector
                    elif item.action == "replace" and item.vector is not None:
                        activation = np.broadcast_to(item.vector, activation.shape).copy()
            return activation

        return hook

    def run_counterfactual_comparison(
        self,
        model,
        input_ids: np.ndarray,
        tokenizer_decoder=None,
    ) -> Dict[str, Any]:
        """
        Runs baseline inference followed by patched counterfactual inference,
        generating a detailed teardown of how internal states and token choices shifted.
        """
        # Baseline run
        base_logits, _, base_trace = model.forward(input_ids, patch_hook=None)

        # Patched run
        patch_hook = self.build_hook()
        patch_logits, _, patch_trace = model.forward(input_ids, patch_hook=patch_hook)

        # Compare output distributions at last token
        last_base_logits = base_logits[0, -1]
        last_patch_logits = patch_logits[0, -1]

        def get_top(logits_arr):
            max_l = np.max(logits_arr)
            exp_l = np.exp(logits_arr - max_l)
            probs = exp_l / (np.sum(exp_l) + 1e-12)
            top_ids = np.argsort(-probs)[:5]
            return [
                {
                    "token_id": int(tid),
                    "token_str": tokenizer_decoder([int(tid)]) if tokenizer_decoder else f"T_{tid}",
                    "prob": round(float(probs[tid]), 4),
                    "logit": round(float(logits_arr[tid]), 4),
                }
                for tid in top_ids
            ]

        base_top = get_top(last_base_logits)
        patch_top = get_top(last_patch_logits)

        # Layer-by-layer cosine similarity delta in residual stream
        residual_divergence = []
        for l_idx in range(len(base_trace["residual_streams"])):
            h_base = base_trace["residual_streams"][l_idx][0, -1]
            h_patch = patch_trace["residual_streams"][l_idx][0, -1]

            norm_base = np.linalg.norm(h_base) + 1e-12
            norm_patch = np.linalg.norm(h_patch) + 1e-12
            cos_sim = float(np.dot(h_base, h_patch) / (norm_base * norm_patch))
            l2_dist = float(np.linalg.norm(h_base - h_patch))

            layer_name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"
            residual_divergence.append({
                "layer_idx": l_idx,
                "layer_name": layer_name,
                "cosine_similarity": round(cos_sim, 4),
                "l2_distance": round(l2_dist, 4),
            })

        return {
            "num_interventions": len(self.interventions),
            "baseline_top_candidates": base_top,
            "patched_top_candidates": patch_top,
            "residual_divergence": residual_divergence,
            "primary_prediction_changed": base_top[0]["token_id"] != patch_top[0]["token_id"],
        }
