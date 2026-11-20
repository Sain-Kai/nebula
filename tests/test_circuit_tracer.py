import numpy as np
import pytest
from nebula.config import ModelConfig
from nebula.core.model import NebulaTransformer
from nebula.core.tokenizer import NebulaTokenizer
from nebula.interpretability.circuit_tracer import CircuitTracer


def test_direct_logit_attribution():
    cfg = ModelConfig(
        vocab_size=320,
        d_model=64,
        n_layers=3,
        n_heads=2,
        d_ff=128,
        is_moe=True,
        moe_layers=[1],
    )
    model = NebulaTransformer(cfg)
    tok = NebulaTokenizer()

    prompt = "Circuit attribution test"
    input_ids = np.array([tok.encode(prompt)])
    logits, _, trace = model.forward(input_ids)

    tracer = CircuitTracer(model, tokenizer_decoder=tok.decode)
    dla = tracer.compute_direct_logit_attribution(
        residual_streams=trace["residual_streams"],
        layer_traces=trace["layer_traces"],
        target_token_id=65,  # 'A'
    )

    assert dla.target_token_id == 65
    assert len(dla.layer_attributions) == 3
    assert len(dla.top_promoting_heads) > 0
    assert len(dla.top_suppressing_heads) > 0
    # Reconstruction error should be minimal
    assert dla.reconstruction_error < 0.5


def test_inter_head_composition():
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model = NebulaTransformer(cfg)
    tok = NebulaTokenizer()

    input_ids = np.array([tok.encode("Composition")])
    _, _, trace = model.forward(input_ids)

    tracer = CircuitTracer(model)
    circuits = tracer.trace_inter_head_composition(trace["layer_traces"])

    assert len(circuits) == 1
    assert "q_composition_strength" in circuits[0]
