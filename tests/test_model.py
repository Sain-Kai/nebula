import numpy as np
import pytest
from nebula.config import ModelConfig
from nebula.core.model import NebulaTransformer


def test_transformer_forward_shape():
    cfg = ModelConfig(
        vocab_size=320,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        is_moe=False,
    )
    model = NebulaTransformer(cfg)
    input_ids = np.array([[10, 20, 30]])
    logits, past_kv, trace = model.forward(input_ids)

    assert logits.shape == (1, 3, 320)
    assert len(past_kv) == 2
    assert len(trace["residual_streams"]) == 3  # embedding + 2 layers


def test_transformer_kv_caching():
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model = NebulaTransformer(cfg)

    # Step 1: prefill 2 tokens
    ids_step1 = np.array([[10, 20]])
    logits1, kv1, _ = model.forward(ids_step1)

    # Step 2: decode 1 token with kv
    ids_step2 = np.array([[30]])
    logits2, kv2, _ = model.forward(ids_step2, past_kv=kv1)

    assert kv2[0][0].shape[2] == 3  # total 3 cached tokens
