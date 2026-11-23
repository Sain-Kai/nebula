import pytest
from nebula.config import ModelConfig
from nebula.core.model import NebulaTransformer
from nebula.interpretability.induction_detector import InductionHeadDetector


def test_induction_head_detector():
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=3, n_heads=2, d_ff=128)
    model = NebulaTransformer(cfg)

    detector = InductionHeadDetector(model, induction_threshold=0.1)
    summary = detector.evaluate_induction_heads(seq_len=16)

    assert summary.total_heads_evaluated == 6  # 3 layers * 2 heads
    assert len(summary.top_induction_heads) <= 6
    assert len(summary.circuit_adjacency) == 2
    assert summary.top_induction_heads[0].induction_score >= 0.0
