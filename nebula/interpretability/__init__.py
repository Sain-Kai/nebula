"""
Mechanistic interpretability tools for KV cache, logit lens, attention heads, and neuron filtering.
"""

from nebula.interpretability.kv_cache import KVCacheTracker, CacheSnapshot
from nebula.interpretability.logit_lens import LogitLens
from nebula.interpretability.attention_analyzer import AttentionAnalyzer
from nebula.interpretability.neuron_filter import NeuronFilter
from nebula.interpretability.circuit_tracer import (
    CircuitTracer,
    DLAResult,
    HeadAttribution,
    LayerComponentAttribution,
)
from nebula.interpretability.sae import (
    SparseAutoencoder,
    ActiveFeature,
    SAETeardownResult,
)

__all__ = [
    "KVCacheTracker",
    "CacheSnapshot",
    "LogitLens",
    "AttentionAnalyzer",
    "NeuronFilter",
    "CircuitTracer",
    "DLAResult",
    "HeadAttribution",
    "LayerComponentAttribution",
    "SparseAutoencoder",
    "ActiveFeature",
    "SAETeardownResult",
]
