"""
Full Transformer architecture with MoE integration and comprehensive state hooks.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np

from nebula.config import ModelConfig
from nebula.core.layers import LayerNorm, MultiHeadAttention, FeedForward
from nebula.core.moe import MoELayer


class TransformerBlock:
    """A single Transformer block containing Attention, Normalization, and MLP or MoE."""

    def __init__(
        self,
        layer_idx: int,
        config: ModelConfig,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.layer_idx = layer_idx
        self.config = config
        self.is_moe = config.is_moe and (layer_idx in config.moe_layers)

        self.ln1 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            use_bias=config.use_bias,
            rng=rng,
        )
        self.ln2 = LayerNorm(config.d_model, eps=config.layer_norm_eps)

        if self.is_moe:
            self.ffn = MoELayer(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                activation_fn=config.activation_fn,
                use_bias=config.use_bias,
                rng=rng,
            )
        else:
            self.ffn = FeedForward(
                d_model=config.d_model,
                d_ff=config.d_ff,
                activation_fn=config.activation_fn,
                use_bias=config.use_bias,
                rng=rng,
            )

    def forward(
        self,
        x: np.ndarray,
        past_k: Optional[np.ndarray] = None,
        past_v: Optional[np.ndarray] = None,
        causal_mask: bool = True,
        patch_hook: Optional[Callable[[int, str, np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Executes block forward pass with intervention hooks and state extraction.
        """
        # 1. Pre-attention norm
        norm1_out, ln1_stats = self.ln1.forward(x)

        # 2. Attention computation
        attn_out, cur_k, cur_v, attn_insp = self.attn.forward(
            norm1_out, past_k=past_k, past_v=past_v, causal_mask=causal_mask
        )

        # Optional activation patch after attention
        if patch_hook is not None:
            attn_out = patch_hook(self.layer_idx, "post_attn", attn_out)

        # 3. First Residual Addition
        x_post_attn = x + attn_out

        # 4. Pre-FFN norm
        norm2_out, ln2_stats = self.ln2.forward(x_post_attn)

        # 5. FFN or MoE computation
        ffn_out, ffn_insp = self.ffn.forward(norm2_out)

        # Optional activation patch after FFN
        if patch_hook is not None:
            ffn_out = patch_hook(self.layer_idx, "post_ffn", ffn_out)

        # 6. Second Residual Addition
        x_post_ffn = x_post_attn + ffn_out

        # Optional activation patch on residual stream
        if patch_hook is not None:
            x_post_ffn = patch_hook(self.layer_idx, "residual_stream", x_post_ffn)

        layer_telemetry = {
            "layer_idx": self.layer_idx,
            "is_moe": self.is_moe,
            "residual_in": x,
            "ln1_stats": ln1_stats,
            "attention": attn_insp,
            "residual_mid": x_post_attn,
            "ln2_stats": ln2_stats,
            "ffn_or_moe": ffn_insp,
            "residual_out": x_post_ffn,
        }

        return x_post_ffn, cur_k, cur_v, layer_telemetry


class NebulaTransformer:
    """
    Complete inspectable Transformer model.
    Maintains clean representations of residual streams, attention matrices,
    and supports counterfactual intervention.
    """

    def __init__(self, config: ModelConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Embeddings
        scale_emb = 1.0 / math.sqrt(config.d_model)
        self.w_token_emb = self.rng.randn(config.vocab_size, config.d_model).astype(np.float32) * scale_emb
        self.w_pos_emb = self.rng.randn(config.max_seq_len, config.d_model).astype(np.float32) * scale_emb

        # Transformer Blocks
        self.blocks = [
            TransformerBlock(layer_idx=i, config=config, rng=self.rng)
            for i in range(config.n_layers)
        ]

        # Final LayerNorm & Unembedding
        self.ln_final = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # Weight-tying or separate unembedding
        self.w_unembed = self.rng.randn(config.d_model, config.vocab_size).astype(np.float32) * scale_emb

    def forward(
        self,
        input_ids: np.ndarray,
        past_kv: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        patch_hook: Optional[Callable[[int, str, np.ndarray], np.ndarray]] = None,
        causal_mask: bool = True,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Forward pass with hook collection.
        
        Args:
            input_ids: Array of shape (batch_size, seq_len)
            past_kv: List of (k, v) tuples for each layer
            patch_hook: Callable for counterfactual activation patching
            causal_mask: Whether to apply causal mask
            
        Returns:
            logits: Array of shape (batch_size, seq_len, vocab_size)
            new_past_kv: Updated list of (k, v) tuples
            trace: Comprehensive dictionary containing all layer-by-layer activations
        """
        batch_size, seq_len = input_ids.shape
        start_pos = past_kv[0][0].shape[2] if past_kv is not None and len(past_kv) > 0 and past_kv[0][0].shape[2] > 0 else 0

        # Token + Positional embeddings
        token_embeddings = self.w_token_emb[input_ids]  # (batch, seq, d_model)
        pos_indices = np.arange(start_pos, start_pos + seq_len)
        pos_embeddings = self.w_pos_emb[pos_indices][None, :, :]  # (1, seq, d_model)

        h = token_embeddings + pos_embeddings

        if patch_hook is not None:
            h = patch_hook(-1, "embedding", h)

        residual_streams: List[np.ndarray] = [h.copy()]
        layer_traces: List[Dict[str, Any]] = []
        new_past_kv: List[Tuple[np.ndarray, np.ndarray]] = []

        # Pass through layers
        for layer_idx, block in enumerate(self.blocks):
            pk = past_kv[layer_idx][0] if past_kv is not None else None
            pv = past_kv[layer_idx][1] if past_kv is not None else None

            h, new_k, new_v, telemetry = block.forward(
                h, past_k=pk, past_v=pv, causal_mask=causal_mask, patch_hook=patch_hook
            )
            new_past_kv.append((new_k, new_v))
            layer_traces.append(telemetry)
            residual_streams.append(h.copy())

        # Final LayerNorm
        h_norm, final_norm_stats = self.ln_final.forward(h)

        # Unembedding: Residual state -> Vocab logits
        logits = np.matmul(h_norm, self.w_unembed)

        full_trace = {
            "input_ids": input_ids,
            "token_embeddings": token_embeddings,
            "pos_embeddings": pos_embeddings,
            "residual_streams": residual_streams,
            "layer_traces": layer_traces,
            "final_norm": final_norm_stats,
            "final_hidden": h_norm,
            "logits": logits,
        }

        return logits, new_past_kv, full_trace
