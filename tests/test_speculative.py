import numpy as np
import pytest
from nebula.config import ModelConfig
from nebula.core.model import NebulaTransformer
from nebula.core.tokenizer import NebulaTokenizer
from nebula.core.speculative import SpeculativeVerifier


def test_speculative_verification():
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model = NebulaTransformer(cfg)
    tok = NebulaTokenizer()

    verifier = SpeculativeVerifier(model, tok)

    prefix_ids = tok.encode("Hello world")
    draft_ids = [65, 66, 67]
    draft_probs = [0.8, 0.7, 0.6]

    trace = verifier.verify_speculative_candidates(prefix_ids, draft_ids, draft_probs)

    assert trace.proposed_tokens == 3
    assert 0 <= trace.accepted_tokens <= 3
    assert len(trace.decisions) >= 1
    assert trace.speedup_ratio > 0.0
