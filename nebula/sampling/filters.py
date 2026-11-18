"""
Response filtering, sampling algorithms, and logit manipulation telemetry.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class LogitFilter:
    """
    Applies temperature scaling, nucleus (top-p), top-k, repetition penalties,
    and logit bias to shape model output distributions.
    Provides transparent telemetry detailing which candidate tokens were filtered out.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ):
        self.temperature = max(temperature, 1e-5)
        self.top_k = max(top_k, 0)
        self.top_p = min(max(top_p, 0.0), 1.0)
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logit_bias = logit_bias or {}

    def process_logits(
        self,
        logits: np.ndarray,
        generated_tokens: Optional[List[int]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Filters raw logits, converts to probabilities, samples next token,
        and returns detailed filtering telemetry.
        
        Args:
            logits: 1D array of shape (vocab_size,)
            generated_tokens: Previously generated token IDs for penalty calculations
            rng: Random state for deterministic sampling
        """
        processed = logits.copy().astype(np.float64)
        vocab_size = len(processed)
        generated_tokens = generated_tokens or []

        # 1. Repetition, frequency, and presence penalties
        token_counts: Dict[int, int] = {}
        for tid in generated_tokens:
            token_counts[tid] = token_counts.get(tid, 0) + 1

        for tid, count in token_counts.items():
            if 0 <= tid < vocab_size:
                # Repetition penalty (multiplicative)
                if self.repetition_penalty != 1.0:
                    if processed[tid] > 0:
                        processed[tid] /= self.repetition_penalty
                    else:
                        processed[tid] *= self.repetition_penalty

                # Frequency penalty (additive per occurrence)
                processed[tid] -= count * self.frequency_penalty

                # Presence penalty (additive once)
                processed[tid] -= self.presence_penalty

        # 2. Logit Bias
        for tid, bias in self.logit_bias.items():
            if 0 <= tid < vocab_size:
                processed[tid] += bias

        # 3. Temperature scaling
        scaled_logits = processed / self.temperature

        # 4. Top-K filtering
        sorted_indices = np.argsort(-scaled_logits)
        filtered_by_top_k: List[int] = []
        if self.top_k > 0 and self.top_k < vocab_size:
            cutoff_indices = sorted_indices[self.top_k :]
            filtered_by_top_k = cutoff_indices[:10].tolist()
            scaled_logits[cutoff_indices] = -1e9

        # 5. Top-P (Nucleus) filtering
        filtered_by_top_p: List[int] = []
        if self.top_p < 1.0:
            # Softmax to check cumulative mass
            max_val = np.max(scaled_logits)
            exp_vals = np.exp(scaled_logits - max_val)
            probs_temp = exp_vals / np.sum(exp_vals)

            sorted_probs = probs_temp[sorted_indices]
            cum_probs = np.cumsum(sorted_probs)

            # Mask tokens beyond cumulative probability threshold
            mask = cum_probs > self.top_p
            # Keep at least the highest probability token
            mask[1:] = mask[:-1]
            mask[0] = False

            masked_indices = sorted_indices[mask]
            filtered_by_top_p = masked_indices[:10].tolist()
            scaled_logits[masked_indices] = -1e9

        # Final Softmax
        max_final = np.max(scaled_logits)
        exp_final = np.exp(scaled_logits - max_final)
        final_probs = exp_final / (np.sum(exp_final) + 1e-12)

        # Sampling
        if rng is None:
            rng = np.random.RandomState(42)

        sampled_token = int(rng.choice(vocab_size, p=final_probs))

        # Top 5 candidate probabilities after all filters
        top_cand_ids = np.argsort(-final_probs)[:5]
        top_candidates = [
            {"token_id": int(tid), "prob": round(float(final_probs[tid]), 4)}
            for tid in top_cand_ids
            if final_probs[tid] > 1e-4
        ]

        telemetry = {
            "temperature": self.temperature,
            "top_k_threshold": self.top_k,
            "top_p_threshold": self.top_p,
            "filtered_by_top_k_count": max(0, vocab_size - self.top_k) if self.top_k > 0 else 0,
            "filtered_by_top_p_count": len(filtered_by_top_p),
            "top_candidates_post_filter": top_candidates,
            "sampled_token": sampled_token,
            "sampled_prob": round(float(final_probs[sampled_token]), 4),
        }

        return sampled_token, telemetry
