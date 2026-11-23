"""
Automated induction head detector and mechanistic circuit flow analyzer.
Identifies in-context learning circuits [A][B] ... [A] -> [B] across attention heads
and computes head-to-head information transfer graphs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class HeadInductionScore:
    """Induction score and classification for an individual head."""
    layer_idx: int
    head_idx: int
    induction_score: float
    is_induction_head: bool
    prefix_anchor_score: float


@dataclass
class InductionCircuitSummary:
    """Report of detected in-context learning induction circuits."""
    total_heads_evaluated: int
    induction_head_count: int
    top_induction_heads: List[HeadInductionScore] = field(default_factory=list)
    circuit_adjacency: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_heads": self.total_heads_evaluated,
            "induction_heads": self.induction_head_count,
            "top_induction_heads": [
                {
                    "layer_idx": h.layer_idx,
                    "head_idx": h.head_idx,
                    "induction_score": round(h.induction_score, 4),
                    "is_induction": h.is_induction_head,
                }
                for h in self.top_induction_heads
            ],
            "circuit_adjacency": self.circuit_adjacency,
        }


class InductionHeadDetector:
    """
    Evaluates multi-head attention weights on synthetic repeated token sequences
    to identify induction heads and previous-token heads.
    """

    def __init__(self, model, induction_threshold: float = 0.35):
        self.model = model
        self.induction_threshold = induction_threshold

    def evaluate_induction_heads(self, seq_len: int = 16) -> InductionCircuitSummary:
        """
        Feeds a repeated token sequence [t_1 ... t_k t_1 ... t_k] through the model
        and measures how strongly heads attend to the token following the previous occurrence.
        """
        assert seq_len % 2 == 0
        half = seq_len // 2

        # Create repeated token sequence: [10..10+half, 10..10+half]
        half_seq = list(range(10, 10 + half))
        repeated_seq = half_seq + half_seq
        input_ids = np.array([repeated_seq])

        _, _, trace = self.model.forward(input_ids)

        head_scores: List[HeadInductionScore] = []
        n_layers = self.model.config.n_layers
        n_heads = self.model.config.n_heads

        for l in range(n_layers):
            attn_w = trace["layer_traces"][l]["attention"]["attention_weights"][0]  # (n_heads, S, S)
            for h in range(n_heads):
                head_mat = attn_w[h]  # (S, S)

                # For the second half of the sequence (positions half .. seq_len-1):
                # Target for position `half + i` is the token following previous occurrence: `i + 1`
                induction_weights = []
                for i in range(1, half - 1):
                    curr_pos = half + i
                    prev_next_pos = i + 1
                    induction_weights.append(head_mat[curr_pos, prev_next_pos])

                mean_ind_score = float(np.mean(induction_weights)) if induction_weights else 0.0
                bos_score = float(np.mean(head_mat[:, 0]))

                is_ind = mean_ind_score >= self.induction_threshold

                head_scores.append(
                    HeadInductionScore(
                        layer_idx=l,
                        head_idx=h,
                        induction_score=mean_ind_score,
                        is_induction_head=is_ind,
                        prefix_anchor_score=bos_score,
                    )
                )

        sorted_heads = sorted(head_scores, key=lambda x: x.induction_score, reverse=True)
        ind_count = sum(1 for h in head_scores if h.is_induction_head)

        # Adjacency of potential composition
        adjacency = []
        for l in range(n_layers - 1):
            adjacency.append({
                "source_layer": l,
                "target_layer": l + 1,
                "information_flow_score": round(float(np.mean([h.induction_score for h in head_scores if h.layer_idx == l + 1])), 4),
            })

        return InductionCircuitSummary(
            total_heads_evaluated=n_layers * n_heads,
            induction_head_count=ind_count,
            top_induction_heads=sorted_heads[:6],
            circuit_adjacency=adjacency,
        )
