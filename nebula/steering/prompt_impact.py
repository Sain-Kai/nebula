"""
Prompt impact analyzer evaluating how instructions, framing, and keywords alter representations.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np


class PromptImpactAnalyzer:
    """
    Evaluates the comparative impact of different prompt styles:
    - Direct vs Few-shot vs Chain of Thought ('Let's think step by step')
    - Sensitive keyword substitutions
    - System prompt framing
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compare_prompts(
        self,
        prompt_a: str,
        prompt_b: str,
        label_a: str = "Prompt A",
        label_b: str = "Prompt B",
    ) -> Dict[str, Any]:
        """
        Runs both prompts through the transformer and compares residual streams,
        attention allocation, and expert routing decisions.
        """
        tokens_a = self.tokenizer.encode(prompt_a)
        tokens_b = self.tokenizer.encode(prompt_b)

        input_a = np.array([tokens_a])
        input_b = np.array([tokens_b])

        logits_a, _, trace_a = self.model.forward(input_a)
        logits_b, _, trace_b = self.model.forward(input_b)

        # Output predictions
        def top_preds(logits_arr):
            last_l = logits_arr[0, -1]
            max_l = np.max(last_l)
            exp_l = np.exp(last_l - max_l)
            probs = exp_l / (np.sum(exp_l) + 1e-12)
            top_ids = np.argsort(-probs)[:3]
            return [
                {
                    "token_id": int(tid),
                    "token_str": self.tokenizer.decode([int(tid)]),
                    "probability": round(float(probs[tid]), 4),
                }
                for tid in top_ids
            ]

        preds_a = top_preds(logits_a)
        preds_b = top_preds(logits_b)

        # Residual stream divergence comparison across layers
        num_layers = len(trace_a["layer_traces"])
        layer_divergence = []

        for l in range(num_layers + 1):
            # Compare representation of the final token in each sequence
            h_a = trace_a["residual_streams"][l][0, -1]
            h_b = trace_b["residual_streams"][l][0, -1]

            norm_a = np.linalg.norm(h_a) + 1e-12
            norm_b = np.linalg.norm(h_b) + 1e-12
            cos_sim = float(np.dot(h_a, h_b) / (norm_a * norm_b))
            l2_dist = float(np.linalg.norm(h_a - h_b))

            layer_name = "Embedding" if l == 0 else f"Layer {l}"
            layer_divergence.append({
                "layer_idx": l,
                "layer_name": layer_name,
                "cosine_similarity": round(cos_sim, 4),
                "l2_distance": round(l2_dist, 4),
            })

        # Expert routing comparison for MoE layers
        moe_comparisons = []
        for l_idx, (lt_a, lt_b) in enumerate(zip(trace_a["layer_traces"], trace_b["layer_traces"])):
            if lt_a["is_moe"] and lt_b["is_moe"]:
                experts_a = lt_a["ffn_or_moe"]["top_k_indices"][0, -1].tolist()
                experts_b = lt_b["ffn_or_moe"]["top_k_indices"][0, -1].tolist()
                weights_a = lt_a["ffn_or_moe"]["top_k_weights"][0, -1].tolist()
                weights_b = lt_b["ffn_or_moe"]["top_k_weights"][0, -1].tolist()

                agreed = set(experts_a) == set(experts_b)
                moe_comparisons.append({
                    "layer_idx": l_idx,
                    "experts_prompt_a": experts_a,
                    "weights_prompt_a": [round(w, 4) for w in weights_a],
                    "experts_prompt_b": experts_b,
                    "weights_prompt_b": [round(w, 4) for w in weights_b],
                    "routing_diverged": not agreed,
                })

        return {
            "prompt_a": {"label": label_a, "text": prompt_a, "tokens": len(tokens_a), "top_predictions": preds_a},
            "prompt_b": {"label": label_b, "text": prompt_b, "tokens": len(tokens_b), "top_predictions": preds_b},
            "layer_divergence": layer_divergence,
            "moe_routing_differences": moe_comparisons,
        }
