import numpy as np
import pytest
from nebula.config import ModelConfig
from nebula.core.model import NebulaTransformer
from nebula.steering.patching import ActivationPatcher, PatchIntervention
from nebula.steering.prompt_impact import PromptImpactAnalyzer
from nebula.core.tokenizer import NebulaTokenizer


def test_activation_patching():
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model = NebulaTransformer(cfg)

    patcher = ActivationPatcher()
    patcher.add_intervention(
        PatchIntervention(layer_idx=0, target="post_attn", action="zero_ablate")
    )

    input_ids = np.array([[10, 20, 30]])
    comp = patcher.run_counterfactual_comparison(model, input_ids)

    assert comp["num_interventions"] == 1
    assert len(comp["residual_divergence"]) == 3


def test_prompt_impact_analyzer():
    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=2, d_ff=128, is_moe=True, moe_layers=[1])
    model = NebulaTransformer(cfg)
    tok = NebulaTokenizer()

    analyzer = PromptImpactAnalyzer(model, tok)
    res = analyzer.compare_prompts("Query A", "Query B")

    assert "layer_divergence" in res
    assert len(res["layer_divergence"]) == 3
