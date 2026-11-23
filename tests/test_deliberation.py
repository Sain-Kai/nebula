import numpy as np
import pytest
from nebula.core.tokenizer import NebulaTokenizer
from nebula.cognitive.deliberation import CognitiveEvaluator


def test_cognitive_evaluator_trace():
    tok = NebulaTokenizer()
    evaluator = CognitiveEvaluator(tok)

    # Simulate sequence of tokens: '?', 'A', 'B'
    tokens = [ord('?'), ord('A'), ord('B')]
    logits_history = [
        np.random.randn(320).astype(np.float32) for _ in range(3)
    ]

    trace = evaluator.evaluate_generation_trace(tokens, logits_history)

    assert trace.total_tokens == 3
    assert len(trace.steps) == 3
    assert trace.cognitive_certainty in ("High Certainty", "Moderate Deliberation", "Deep Uncertainty")
    assert trace.steps[0].is_deliberation_token == True
    assert 0.0 <= trace.deliberation_ratio <= 1.0
