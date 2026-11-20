import numpy as np
import pytest
from nebula.config import ModelConfig
from nebula.core.model import NebulaTransformer
from nebula.core.tokenizer import NebulaTokenizer
from nebula.core.backward import BackwardPassEngine


def test_backward_pass_gradients():
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

    prompt = "Test gradient flow"
    input_ids = np.array([tok.encode(prompt)])
    target_token_id = 65  # ASCII 'A'

    engine = BackwardPassEngine(model, tokenizer=tok)
    telemetry = engine.compute_loss_and_gradients(input_ids, target_token_id)

    assert telemetry.loss > 0.0
    assert len(telemetry.layer_gradients) == 3
    assert telemetry.embedding_grad_norm > 0.0
    assert telemetry.unembed_grad_norm > 0.0
    assert telemetry.gradient_health in ("Healthy", "Vanishing", "Exploding")
    assert telemetry.target_token_id == target_token_id
