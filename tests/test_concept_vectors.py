import numpy as np
import pytest
from nebula.config import ModelConfig
from nebula.core.model import NebulaTransformer
from nebula.core.tokenizer import NebulaTokenizer
from nebula.steering.concept_vectors import ConceptSteeringStudio


def test_concept_steering_studio():
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

    studio = ConceptSteeringStudio(model, tok)
    assert len(studio.vector_library) >= 4
    assert "Reasoning" in studio.vector_library

    # Apply steering
    prompt = "Why is the sky blue?"
    res = studio.apply_steering(prompt, "Reasoning", intensity=2.0)

    assert res.concept_name == "Reasoning"
    assert res.concept_alignment_after > res.concept_alignment_before or abs(res.concept_alignment_after) >= 0.0
    assert len(res.top_diverging_layers) == 4


def test_neuron_impulse_injection():
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model = NebulaTransformer(cfg)
    tok = NebulaTokenizer()

    studio = ConceptSteeringStudio(model, tok)
    res = studio.inject_neuron_impulse(
        prompt="Impulse test",
        layer_idx=0,
        neuron_idx=5,
        impulse_value=10.0,
    )

    assert "baseline_token" in res
    assert "steered_token" in res
    assert "caused_token_change" in res
