import numpy as np
import pytest
from nebula.interpretability.kv_cache import KVCacheTracker
from nebula.interpretability.logit_lens import LogitLens
from nebula.interpretability.attention_analyzer import AttentionAnalyzer
from nebula.interpretability.neuron_filter import NeuronFilter


def test_kv_cache_tracker():
    tracker = KVCacheTracker(n_layers=2, n_heads=2, d_head=32)
    fake_k = np.zeros((1, 2, 5, 32), dtype=np.float32)
    fake_v = np.zeros((1, 2, 5, 32), dtype=np.float32)
    past_kv = [(fake_k, fake_v), (fake_k, fake_v)]

    snap = tracker.record_step(0, past_kv)
    assert snap.total_cached_tokens == 5
    assert snap.memory_bytes > 0


def test_logit_lens():
    w_unembed = np.random.randn(64, 320).astype(np.float32)
    lens = LogitLens(w_unembed)

    h0 = np.random.randn(1, 3, 64).astype(np.float32)
    h1 = np.random.randn(1, 3, 64).astype(np.float32)
    res = lens.analyze_residual_stream([h0, h1], token_pos=-1)

    assert len(res) == 2
    assert "top_prediction" in res[0]
    assert "entropy" in res[0]


def test_attention_analyzer():
    analyzer = AttentionAnalyzer(n_heads=2)
    fake_attn = np.ones((1, 2, 4, 4), dtype=np.float32) / 4.0
    report = analyzer.analyze_layer_attention(fake_attn)

    assert len(report["head_analyses"]) == 2
    assert "mean_entropy" in report["head_analyses"][0]


def test_neuron_filter():
    filter_mod = NeuronFilter()
    acts = np.random.randn(1, 4, 128).astype(np.float32)
    res = filter_mod.analyze_layer_neurons(acts, layer_idx=0)

    assert res["total_neurons"] == 128
    assert len(res["top_neurons"]) > 0
