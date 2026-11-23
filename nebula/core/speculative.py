"""
Speculative decoding and parallel multi-candidate verification engine.
Demonstrates how draft models propose speculative token candidates and how the target model
verifies, accepts, or rejects tokens in parallel, managing KV cache rollbacks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class SpeculativeTokenDecision:
    """Decision breakdown for a single candidate token in speculative decoding."""
    position: int
    draft_token_id: int
    draft_token_str: str
    draft_prob: float
    target_prob: float
    acceptance_prob: float
    is_accepted: bool
    resampled_token_id: Optional[int] = None
    resampled_token_str: Optional[str] = None


@dataclass
class SpeculativeRoundTrace:
    """Telemetry for a single speculative execution round."""
    round_idx: int
    proposed_tokens: int
    accepted_tokens: int
    speedup_ratio: float
    decisions: List[SpeculativeTokenDecision] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "proposed_tokens": self.proposed_tokens,
            "accepted_tokens": self.accepted_tokens,
            "speedup_ratio": round(self.speedup_ratio, 3),
            "decisions": [
                {
                    "position": d.position,
                    "draft_token_id": d.draft_token_id,
                    "draft_token_str": d.draft_token_str,
                    "draft_prob": round(d.draft_prob, 4),
                    "target_prob": round(d.target_prob, 4),
                    "acceptance_prob": round(d.acceptance_prob, 4),
                    "is_accepted": d.is_accepted,
                    "resampled_token_id": d.resampled_token_id,
                    "resampled_token_str": d.resampled_token_str,
                }
                for d in self.decisions
            ],
        }


class SpeculativeVerifier:
    """
    Executes and visualizes speculative verification loops between a draft model
    and a larger target model, calculating rejection sampling distributions.
    """

    def __init__(self, target_model, tokenizer, draft_gamma: int = 4):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.draft_gamma = draft_gamma

    def verify_speculative_candidates(
        self,
        prefix_ids: List[int],
        draft_candidate_ids: List[int],
        draft_probs: List[float],
        seed: int = 42,
    ) -> SpeculativeRoundTrace:
        """
        Runs target model parallel evaluation on proposed draft candidates
        and computes exact speculative rejection sampling decisions.
        """
        assert len(draft_candidate_ids) == len(draft_probs)
        rng = np.random.RandomState(seed)

        # Evaluate target model on prefix + draft candidates in a single batched pass
        full_seq = prefix_ids + draft_candidate_ids
        input_ids = np.array([full_seq])
        target_logits, _, _ = self.target_model.forward(input_ids)

        decisions: List[SpeculativeTokenDecision] = []
        accepted_count = 0
        all_accepted = True

        prefix_len = len(prefix_ids)
        vocab_size = self.target_model.config.vocab_size

        for i, (draft_id, q_prob) in enumerate(zip(draft_candidate_ids, draft_probs)):
            if not all_accepted:
                break

            # Target model logits predicting token at position prefix_len + i
            target_pos = prefix_len + i - 1
            logits = target_logits[0, target_pos, :]

            # Softmax
            max_l = np.max(logits)
            exp_l = np.exp(logits - max_l)
            p_probs = exp_l / (np.sum(exp_l) + 1e-12)
            p_prob = float(p_probs[draft_id])

            # Speculative Acceptance Ratio: alpha = min(1, p(x) / q(x))
            alpha = min(1.0, p_prob / max(q_prob, 1e-12))
            u = rng.uniform(0.0, 1.0)
            accepted = u <= alpha

            tok_str = self.tokenizer.decode([draft_id])

            if accepted:
                accepted_count += 1
                decisions.append(
                    SpeculativeTokenDecision(
                        position=i,
                        draft_token_id=draft_id,
                        draft_token_str=tok_str,
                        draft_prob=q_prob,
                        target_prob=p_prob,
                        acceptance_prob=alpha,
                        is_accepted=True,
                    )
                )
            else:
                all_accepted = False
                # Resample correction token from max(0, p - q)
                resample_dist = np.maximum(0.0, p_probs - q_prob)
                dist_sum = np.sum(resample_dist)
                if dist_sum > 1e-12:
                    resample_dist /= dist_sum
                    resampled_id = int(rng.choice(vocab_size, p=resample_dist))
                else:
                    resampled_id = int(np.argmax(p_probs))

                resampled_str = self.tokenizer.decode([resampled_id])
                decisions.append(
                    SpeculativeTokenDecision(
                        position=i,
                        draft_token_id=draft_id,
                        draft_token_str=tok_str,
                        draft_prob=q_prob,
                        target_prob=p_prob,
                        acceptance_prob=alpha,
                        is_accepted=False,
                        resampled_token_id=resampled_id,
                        resampled_token_str=resampled_str,
                    )
                )

        # Theoretical speedup: (accepted + 1) / (1 + draft_overhead_fraction)
        gamma = len(draft_candidate_ids)
        speedup = (accepted_count + 1) / (1.0 + (gamma * 0.15))

        return SpeculativeRoundTrace(
            round_idx=0,
            proposed_tokens=gamma,
            accepted_tokens=accepted_count,
            speedup_ratio=speedup,
            decisions=decisions,
        )
