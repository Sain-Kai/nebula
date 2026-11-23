import numpy as np
import pytest
from nebula.interpretability.sae import SparseAutoencoder


def test_sparse_autoencoder_decomposition():
    d_model = 64
    sae = SparseAutoencoder(
        d_model=d_model,
        expansion_factor=4,
        k_sparsity=8,
        layer_idx=1,
    )

    x = np.random.randn(1, 4, d_model).astype(np.float32)
    features = sae.encode(x)
    assert features.shape == (1, 4, d_model * 4)

    # Check Top-K sparsity
    last_feat = features[0, -1, :]
    l0 = np.sum(last_feat > 0.0)
    assert l0 <= 8

    res = sae.decompose(x, top_n=5)
    assert res.layer_idx == 1
    assert res.l0_sparsity <= 8
    assert len(res.active_features) <= 5
    assert res.reconstruction_loss >= 0.0
