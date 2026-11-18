"""
Nebula: Interactive LLM Mechanistic Interpretability and Visualization Suite.

Provides fine-grained, neuron-by-neuron, layer-by-layer teardown analysis
of Large Language Model inference dynamics, attention patterns, KV cache lifecycle,
Mixture of Experts routing, logit lens projections, and counterfactual activation patching.
"""

__version__ = "0.1.0"
__author__ = "Sain-Kai"

from nebula.config import NebulaConfig

__all__ = ["NebulaConfig", "__version__"]
