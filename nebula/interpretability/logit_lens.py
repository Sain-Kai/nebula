"""
Logit Lens extractor for residual stream early unembedding and thought progression analysis.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np


class LogitLens:
    """
    Applies the unembedding matrix W_U directly to intermediate residual streams
    to reveal how the model's internal prediction solidifies across layers.
    """

    def __init__(self, w_unembed: np.ndarray, tokenizer_decoder=None):
        self.w_unembed = w_unembed  # (d_model, vocab_size)
        self.tokenizer_decoder = tokenizer_decoder

    def analyze_residual_stream(
        self,
        residual_streams: List[np.ndarray],
        token_pos: int = -1,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extracts top token candidates and entropy for each layer at token position `token_pos`.
        
        Args:
            residual_streams: List of residual stream tensors, length = n_layers + 1.
                              Each tensor has shape (batch, seq_len, d_model).
            token_pos: Index of token to inspect (default: last token).
            top_k: Number of candidate tokens to return per layer.
            
        Returns:
            List of layer prediction dictionaries.
        """
        layer_predictions: List[Dict[str, Any]] = []

        for layer_idx, h in enumerate(residual_streams):
            # Extract representation at chosen token position
            h_token = h[0, token_pos, :]  # (d_model,)

            # Project through unembedding matrix
            logits = np.matmul(h_token, self.w_unembed)  # (vocab_size,)

            # Softmax
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / (np.sum(exp_logits) + 1e-12)

            # Top-k tokens
            top_indices = np.argsort(-probs)[:top_k]
            top_probs = probs[top_indices]

            candidates = []
            for tid, prob in zip(top_indices, top_probs):
                if self.tokenizer_decoder is not None:
                    txt = self.tokenizer_decoder([int(tid)])
                else:
                    txt = f"Token_{tid}"
                candidates.append({
                    "token_id": int(tid),
                    "token_str": txt,
                    "probability": round(float(prob), 4),
                    "logit": round(float(logits[tid]), 4),
                })

            # Calculate predictive entropy
            safe_probs = np.clip(probs, 1e-10, 1.0)
            entropy = float(-np.sum(probs * np.log(safe_probs)))

            layer_name = "Embedding" if layer_idx == 0 else f"Layer {layer_idx}"
            layer_predictions.append({
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "entropy": round(entropy, 4),
                "top_prediction": candidates[0] if candidates else None,
                "top_k_candidates": candidates,
                "norm": round(float(np.linalg.norm(h_token)), 4),
            })

        return layer_predictions

    def detect_prediction_shifts(self, layer_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identifies layers where the top predicted token changed, indicating key decision points.
        """
        shifts: List[Dict[str, Any]] = []
        for i in range(1, len(layer_predictions)):
            prev_pred = layer_predictions[i - 1]["top_prediction"]
            curr_pred = layer_predictions[i]["top_prediction"]

            if prev_pred and curr_pred and prev_pred["token_id"] != curr_pred["token_id"]:
                shifts.append({
                    "from_layer": layer_predictions[i - 1]["layer_name"],
                    "to_layer": layer_predictions[i]["layer_name"],
                    "previous_token": prev_pred["token_str"],
                    "new_token": curr_pred["token_str"],
                    "previous_prob": prev_pred["probability"],
                    "new_prob": curr_pred["probability"],
                })

        return shifts
