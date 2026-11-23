"""
Sparse Autoencoder (SAE) for monosemantic feature extraction and polysemanticity teardown.
Decomposes dense transformer residual stream states into sparse, interpretable semantic latents.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class ActiveFeature:
    """An active monosemantic feature extracted by the Sparse Autoencoder."""
    feature_idx: int
    activation: float
    decoder_norm: float
    semantic_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_idx": self.feature_idx,
            "activation": round(float(self.activation), 4),
            "decoder_norm": round(float(self.decoder_norm), 4),
            "semantic_label": self.semantic_label,
        }


@dataclass
class SAETeardownResult:
    """Results from decomposing an activation vector through the Sparse Autoencoder."""
    layer_idx: int
    l0_sparsity: int
    reconstruction_loss: float
    explained_variance: float
    active_features: List[ActiveFeature] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "l0_sparsity": self.l0_sparsity,
            "reconstruction_loss": round(float(self.reconstruction_loss), 4),
            "explained_variance": round(float(self.explained_variance), 4),
            "active_features": [f.to_dict() for f in self.active_features],
        }


class SparseAutoencoder:
    """
    Top-K Sparse Autoencoder mapping dense model activations to overcomplete monosemantic features:
    f = TopK(ReLU(W_enc (x - b_dec) + b_enc), k)
    x_hat = W_dec f + b_dec
    """

    def __init__(
        self,
        d_model: int = 128,
        expansion_factor: int = 4,
        k_sparsity: int = 16,
        layer_idx: int = 1,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.n_features = d_model * expansion_factor
        self.k_sparsity = min(k_sparsity, self.n_features)
        self.layer_idx = layer_idx

        rng = np.random.RandomState(seed)
        scale = 1.0 / math.sqrt(d_model)

        self.w_enc = rng.randn(d_model, self.n_features).astype(np.float32) * scale
        self.b_enc = np.zeros(self.n_features, dtype=np.float32)

        # Decoder weights initialized and normalized to unit L2 sphere
        self.w_dec = rng.randn(self.n_features, d_model).astype(np.float32) * scale
        norms = np.linalg.norm(self.w_dec, axis=-1, keepdims=True) + 1e-12
        self.w_dec = self.w_dec / norms

        self.b_dec = np.zeros(d_model, dtype=np.float32)
        self.feature_labels: Dict[int, str] = {}

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encodes hidden vector x into sparse feature activations using Top-K truncation.
        x: (..., d_model) -> f: (..., n_features)
        """
        # Center by decoder bias
        x_centered = x - self.b_dec
        pre_act = np.matmul(x_centered, self.w_enc) + self.b_enc
        act = np.maximum(0.0, pre_act)

        # Top-K Sparsity enforcement
        if self.k_sparsity < self.n_features:
            ranks = np.argsort(-act, axis=-1)
            keep_mask = ranks < self.k_sparsity
            act = np.where(keep_mask, act, 0.0)

        return act

    def decode(self, f: np.ndarray) -> np.ndarray:
        """
        Reconstructs original space from feature latents: x_hat = W_dec @ f + b_dec.
        """
        return np.matmul(f, self.w_dec) + self.b_dec

    def decompose(self, x: np.ndarray, top_n: int = 8) -> SAETeardownResult:
        """
        Decomposes activation vector x, computing L0 sparsity and returning top active features.
        """
        # Take last token if sequence
        if x.ndim == 3:
            x_vec = x[0, -1, :]
        elif x.ndim == 2:
            x_vec = x[-1, :]
        else:
            x_vec = x

        features = self.encode(x_vec)
        x_hat = self.decode(features)

        # Reconstruction error
        diff = x_vec - x_hat
        mse = float(np.mean(diff ** 2))

        # Explained variance: 1 - Var(x - x_hat) / Var(x)
        var_x = float(np.var(x_vec)) + 1e-12
        var_diff = float(np.var(diff))
        r2 = max(0.0, min(1.0, 1.0 - (var_diff / var_x)))

        l0 = int(np.sum(features > 0.0))

        # Top active features
        active_indices = np.where(features > 0.0)[0]
        sorted_by_act = active_indices[np.argsort(-features[active_indices])]

        active_list: List[ActiveFeature] = []
        for idx in sorted_by_act[:top_n]:
            label = self.feature_labels.get(int(idx), f"Latent_Feature_{idx}")
            dec_norm = float(np.linalg.norm(self.w_dec[idx]))
            active_list.append(
                ActiveFeature(
                    feature_idx=int(idx),
                    activation=float(features[idx]),
                    decoder_norm=dec_norm,
                    semantic_label=label,
                )
            )

        return SAETeardownResult(
            layer_idx=self.layer_idx,
            l0_sparsity=l0,
            reconstruction_loss=mse,
            explained_variance=r2,
            active_features=active_list,
        )
