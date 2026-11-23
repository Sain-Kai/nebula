"""
Safety guardrail and logit mask filtering pipeline.
Inspects and censors harmful token candidates, suppresses jailbreak vectors,
and computes prompt injection vulnerability metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import numpy as np


@dataclass
class GuardrailTelemetry:
    """Telemetry report describing guardrail interventions on candidate tokens."""
    prompt_injection_score: float
    risk_level: str  # 'Low', 'Medium', 'High'
    banned_tokens_suppressed: List[Dict[str, Any]]
    safety_override_triggered: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_injection_score": round(self.prompt_injection_score, 3),
            "risk_level": self.risk_level,
            "banned_tokens_suppressed": self.banned_tokens_suppressed,
            "safety_override_triggered": self.safety_override_triggered,
        }


class SafetyGuardrailFilter:
    """
    Applies token-level safety masks, prevents prompt injection jailbreaks,
    and inspects which tokens were suppressed by safety policies.
    """

    INJECTION_TRIGGERS = [
        "ignore previous instructions",
        "system override",
        "jailbreak",
        "disregard guidelines",
        "you are now evil",
        "dan mode",
    ]

    def __init__(self, tokenizer, blocked_token_ids: Optional[Set[int]] = None):
        self.tokenizer = tokenizer
        self.blocked_token_ids = blocked_token_ids or set()

    def add_blocked_token(self, token_id: int):
        self.blocked_token_ids.add(token_id)

    def detect_prompt_injection(self, prompt: str) -> float:
        """Heuristic evaluation of jailbreak risk score between 0.0 and 1.0."""
        lower_p = prompt.lower()
        score = 0.0
        for trigger in self.INJECTION_TRIGGERS:
            if trigger in lower_p:
                score += 0.4
        return min(1.0, score)

    def filter_logits(
        self,
        logits: np.ndarray,
        prompt: str,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, GuardrailTelemetry]:
        """
        Masks out blocked or unsafe tokens by setting their logits to -1e9.
        """
        masked_logits = logits.copy()
        suppressed: List[Dict[str, Any]] = []

        injection_score = self.detect_prompt_injection(prompt)
        risk = "High" if injection_score >= threshold else ("Medium" if injection_score > 0.2 else "Low")
        override = False

        for tid in self.blocked_token_ids:
            if 0 <= tid < len(masked_logits):
                raw_l = float(masked_logits[tid])
                if raw_l > -1e4:
                    suppressed.append({
                        "token_id": int(tid),
                        "token_str": self.tokenizer.decode([tid]),
                        "original_logit": round(raw_l, 4),
                    })
                    masked_logits[tid] = -1e9
                    override = True

        return masked_logits, GuardrailTelemetry(
            prompt_injection_score=injection_score,
            risk_level=risk,
            banned_tokens_suppressed=suppressed,
            safety_override_triggered=override,
        )
