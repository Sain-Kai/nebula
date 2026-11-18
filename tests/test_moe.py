import numpy as np
import pytest
from nebula.core.moe import MoERouter, MoELayer


def test_moe_router_topk():
    router = MoERouter(d_model=64, num_experts=4, top_k=2)
    x = np.random.randn(1, 3, 64).astype(np.float32)
    top_indices, top_weights, diag = router.route(x)

    assert top_indices.shape == (1, 3, 2)
    assert top_weights.shape == (1, 3, 2)
    # Weights should sum to 1
    weight_sums = np.sum(top_weights, axis=-1)
    np.testing.assert_allclose(weight_sums, 1.0, atol=1e-5)


def test_moe_layer_forward():
    moe = MoELayer(d_model=64, d_ff=128, num_experts=4, top_k=2)
    x = np.random.randn(1, 4, 64).astype(np.float32)
    out, insp = moe.forward(x)

    assert out.shape == (1, 4, 64)
    assert "router" in insp
    assert "top_k_indices" in insp
