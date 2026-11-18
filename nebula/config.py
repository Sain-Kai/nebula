"""
Configuration specifications for Nebula models, interpretability pipelines, and servers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for a Transformer model architecture in Nebula."""
    vocab_size: int = 320
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    
    # Mixture of Experts parameters
    is_moe: bool = True
    num_experts: int = 4
    top_k_experts: int = 2
    moe_layers: List[int] = field(default_factory=lambda: [1, 3])
    
    # Activation and normalization
    activation_fn: str = "gelu"  # 'gelu', 'relu', 'swiglu'
    use_bias: bool = True
    rope_embeddings: bool = True


@dataclass
class SamplingConfig:
    """Inference sampling parameters and filtering controls."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_new_tokens: int = 32
    seed: Optional[int] = 42


@dataclass
class ServerConfig:
    """Nebula local visualization server settings."""
    host: str = "127.0.0.1"
    port: int = 8765
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    stream_delay_ms: int = 40  # Simulation delay for pedagogical token-by-token observation


@dataclass
class NebulaConfig:
    """Master configuration encompassing model, sampling, and server setups."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
