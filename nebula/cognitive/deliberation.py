"""
Cognitive deliberation and internal reasoning trace evaluator.
Tracks token-by-token entropy variance, hesitation spikes, certainty transitions,
and deliberation spans during chain-of-thought and autoregressive generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class TokenDeliberationStep:
    """Telemetry for an individual token's cognitive evaluation."""
    step_idx: int
    token_id: int
    token_str: str
    probability: float
    entropy: float
    is_hesitation_point: bool
    is_deliberation_token: bool
    cumulative_confidence: float
    top_alternative_token: str
    top_alternative_prob: float


@dataclass
class DeliberationTrace:
    """Complete summary of the cognitive reasoning and deliberation process."""
    total_tokens: int
    mean_entropy: float
    hesitation_count: int
    deliberation_ratio: float
    cognitive_certainty: str  # 'High Certainty', 'Moderate Deliberation', 'Deep Uncertainty'
    steps: List[TokenDeliberationStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "mean_entropy": round(self.mean_entropy, 4),
            "hesitation_count": self.hesitation_count,
            "deliberation_ratio": round(self.deliberation_ratio, 4),
            "cognitive_certainty": self.cognitive_certainty,
            "steps": [
                {
                    "step_idx": s.step_idx,
                    "token_id": s.token_id,
                    "token_str": s.token_str,
                    "probability": round(s.probability, 4),
                    "entropy": round(s.entropy, 4),
                    "is_hesitation_point": s.is_hesitation_point,
                    "is_deliberation_token": s.is_deliberation_token,
                    "cumulative_confidence": round(s.cumulative_confidence, 4),
                    "top_alternative": s.top_alternative_token,
                    "alternative_prob": round(s.top_alternative_prob, 4),
                }
                for s in self.steps
            ],
        }


class CognitiveEvaluator:
    """
    Evaluates the model's internal deliberation dynamics and thought process
    by tracking entropy fluctuations and identifying cognitive pivot points.
    """

    REASONING_KEYWORDS = {
        "because", "therefore", "however", "although", "since",
        "first", "second", "then", "if", "otherwise", "conclude",
        "step", "calculate", "analyze", "consider", "assume"
    }

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def evaluate_generation_trace(
        self,
        token_ids: List[int],
        logits_history: List[np.ndarray],
        entropy_threshold: float = 2.5,
    ) -> DeliberationTrace:
        """
        Analyzes a sequence of emitted tokens and corresponding output logits.
        
        Args:
            token_ids: List of emitted token IDs
            logits_history: List of 1D logits arrays, one per emitted token
            entropy_threshold: Threshold above which a step is tagged as a hesitation point
        """
        assert len(token_ids) == len(logits_history), "token_ids and logits_history must have equal length"

        steps: List[TokenDeliberationStep] = []
        conf_acc = 0.0
        deliberation_token_count = 0
        hesitation_count = 0
        total_entropy = 0.0

        for i, (tid, logits) in enumerate(zip(token_ids, logits_history)):
            # Softmax
            max_l = np.max(logits)
            exp_l = np.exp(logits - max_l)
            probs = exp_l / (np.sum(exp_l) + 1e-12)

            token_p = float(probs[tid])
            conf_acc += token_p

            # Predictive Entropy: -sum p log p
            safe_p = np.clip(probs, 1e-10, 1.0)
            entropy = float(-np.sum(probs * np.log(safe_p)))
            total_entropy += entropy

            # Top alternative token
            ranked_indices = np.argsort(-probs)
            alt_idx = ranked_indices[1] if ranked_indices[0] == tid else ranked_indices[0]
            alt_str = self.tokenizer.decode([int(alt_idx)])
            alt_p = float(probs[alt_idx])

            token_str = self.tokenizer.decode([tid])
            clean_word = token_str.strip().lower()

            # Check if token matches reasoning keywords, stems, or deliberation symbols
            is_reasoning_keyword = (
                clean_word in self.REASONING_KEYWORDS or
                any(clean_word.startswith(kw[:3]) for kw in self.REASONING_KEYWORDS if len(clean_word) >= 3) or
                clean_word in {"->", "=>", "...", "?", ":", "="}
            )
            if is_reasoning_keyword:
                deliberation_token_count += 1

            is_hesitation = entropy > entropy_threshold
            if is_hesitation:
                hesitation_count += 1

            steps.append(
                TokenDeliberationStep(
                    step_idx=i,
                    token_id=tid,
                    token_str=token_str,
                    probability=token_p,
                    entropy=entropy,
                    is_hesitation_point=is_hesitation,
                    is_deliberation_token=is_reasoning_keyword,
                    cumulative_confidence=conf_acc / (i + 1),
                    top_alternative_token=alt_str,
                    top_alternative_prob=alt_p,
                )
            )

        n = len(steps)
        mean_ent = total_entropy / n if n > 0 else 0.0
        delib_ratio = deliberation_token_count / n if n > 0 else 0.0

        if mean_ent < 1.2 and hesitation_count <= 1:
            certainty = "High Certainty"
        elif mean_ent < 2.5:
            certainty = "Moderate Deliberation"
        else:
            certainty = "Deep Uncertainty"

        return DeliberationTrace(
            total_tokens=n,
            mean_entropy=mean_ent,
            hesitation_count=hesitation_count,
            deliberation_ratio=delib_ratio,
            cognitive_certainty=certainty,
            steps=steps,
        )
