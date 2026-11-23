import numpy as np
import pytest
from nebula.core.tokenizer import NebulaTokenizer
from nebula.interpretability.sae import SparseAutoencoder
from nebula.interpretability.feature_catalog import FeatureCatalog


def test_feature_catalog():
    tok = NebulaTokenizer()
    catalog = FeatureCatalog(tok)
    sae = SparseAutoencoder(d_model=64, expansion_factor=2, k_sparsity=4)

    tags = catalog.auto_tag_sae(sae)
    assert len(tags) == sae.n_features
    assert 0 in tags

    w_unembed = np.random.randn(64, 320).astype(np.float32)
    effect = catalog.compute_feature_to_logit_attribution(sae, w_unembed, feature_idx=0, top_k=3)

    assert effect.feature_idx == 0
    assert len(effect.top_promoted_tokens) == 3
    assert len(effect.top_suppressed_tokens) == 3
    assert effect.top_promoted_tokens[0]["logit_effect"] >= effect.top_suppressed_tokens[0]["logit_effect"]
