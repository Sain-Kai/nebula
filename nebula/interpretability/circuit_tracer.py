"""
Direct Logit Attribution (DLA) and mechanistic circuit tracing.
Decomposes token predictions into exact additive contributions from individual attention heads,
MLP/MoE blocks, and input embeddings, exposing the computational circuits that drive decisions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class HeadAttribution:
    """Attribution score of an individual attention head to the target token logit."""
    layer_idx: int
    head_idx: int
    direct_logit_contribution: float
    percentage_contribution: float
    is_positive_promoter: bool


@dataclass
class LayerComponentAttribution:
    """Breakdown of contributions within a layer."""
    layer_idx: int
    is_moe: bool
    attn_total_contribution: float
    ffn_or_moe_contribution: float
    head_attributions: List[HeadAttribution] = field(default_factory=list)


@dataclass
class DLAResult:
    """Full Direct Logit Attribution decomposition."""
    target_token_id: int
    target_token_str: str
    baseline_logit: float
    embedding_contribution: float
    layer_attributions: List[LayerComponentAttribution]
    top_promoting_heads: List[HeadAttribution]
    top_suppressing_heads: List[HeadAttribution]
    reconstructed_logit: float
    reconstruction_error: float


class CircuitTracer:
    """
    Computes Direct Logit Attribution (DLA) and identifies mechanistic circuits
    connecting attention heads and MLP blocks to model outputs.
    """

    def __init__(self, model, tokenizer_decoder=None):
        self.model = model
        self.tokenizer_decoder = tokenizer_decoder

    def compute_direct_logit_attribution(
        self,
        residual_streams: List[np.ndarray],
        layer_traces: List[Dict[str, Any]],
        target_token_id: int,
        token_pos: int = -1,
    ) -> DLAResult:
        """
        Decomposes the target token's logit into the sum of:
        - Embedding vector contribution
        - Attention output contribution per head
        - MLP / MoE output contribution per layer
        
        Args:
            residual_streams: List of residual stream representations (length L + 1)
            layer_traces: Telemetry traces from each layer forward pass
            target_token_id: The token whose logit is being decomposed
            token_pos: Sequence position to inspect (default: last token)
        """
        w_unembed = self.model.w_unembed  # (d_model, vocab_size)
        unembed_col = w_unembed[:, target_token_id]  # (d_model,)

        # 1. Embedding contribution: h_0 @ W_U[:, target]
        h_0 = residual_streams[0][0, token_pos, :]
        emb_contrib = float(np.dot(h_0, unembed_col))

        total_contrib = emb_contrib
        layer_attributions: List[LayerComponentAttribution] = []
        all_heads: List[HeadAttribution] = []

        d_model = self.model.config.d_model
        n_heads = self.model.config.n_heads
        d_head = d_model // n_heads

        # 2. Iterate through layers and decompose Attn & FFN contributions
        for l_idx, lt in enumerate(layer_traces):
            is_moe = lt["is_moe"]
            block = self.model.blocks[l_idx]

            # Attention contribution:
            # We decompose W_O into head-specific sub-matrices:
            # attn_out = sum_h (head_out_h @ W_O[h*d_head : (h+1)*d_head, :])
            attn_weights = lt["attention"]["attention_weights"][0]  # (n_heads, seq_len, total_kv)
            v_cache = lt["attention"]["value"][0]  # (n_heads, total_kv, d_head)
            w_o = block.attn.w_o  # (d_model, d_model)

            head_attributions: List[HeadAttribution] = []
            layer_attn_total = 0.0

            for h in range(n_heads):
                w_o_head = w_o[h * d_head : (h + 1) * d_head, :]  # (d_head, d_model)
                attn_row = attn_weights[h, token_pos, :]  # (total_kv,)
                head_v = v_cache[h]  # (total_kv, d_head)

                # Weighted value vector for this head
                head_out_vec = np.dot(attn_row, head_v)  # (d_head,)
                head_proj_out = np.dot(head_out_vec, w_o_head)  # (d_model,)

                # Direct attribution of this head to target logit
                head_contrib = float(np.dot(head_proj_out, unembed_col))
                layer_attn_total += head_contrib

                head_attributions.append(
                    HeadAttribution(
                        layer_idx=l_idx,
                        head_idx=h,
                        direct_logit_contribution=round(head_contrib, 4),
                        percentage_contribution=0.0,  # Computed below
                        is_positive_promoter=head_contrib > 0.0,
                    )
                )
                all_heads.append(head_attributions[-1])

            # FFN / MoE contribution:
            # difference in residual stream: delta_ffn = residual_out - residual_mid
            res_mid = lt["residual_mid"][0, token_pos, :]
            res_out = lt["residual_out"][0, token_pos, :]
            ffn_vec = res_out - res_mid  # (d_model,)
            ffn_contrib = float(np.dot(ffn_vec, unembed_col))

            total_contrib += layer_attn_total + ffn_contrib

            layer_attributions.append(
                LayerComponentAttribution(
                    layer_idx=l_idx,
                    is_moe=is_moe,
                    attn_total_contribution=round(layer_attn_total, 4),
                    ffn_or_moe_contribution=round(ffn_contrib, 4),
                    head_attributions=head_attributions,
                )
            )

        # Baseline logit from final layer
        h_final = residual_streams[-1][0, token_pos, :]
        baseline_logit = float(np.dot(h_final, unembed_col))

        # Calculate percentage contributions of each head relative to abs sum
        total_abs = sum(abs(h.direct_logit_contribution) for h in all_heads) + 1e-12
        for h in all_heads:
            h.percentage_contribution = round((h.direct_logit_contribution / total_abs) * 100.0, 2)

        # Rank heads by direct logit contribution
        sorted_heads = sorted(all_heads, key=lambda x: x.direct_logit_contribution, reverse=True)
        top_promoters = sorted_heads[:5]
        top_suppressors = sorted(all_heads, key=lambda x: x.direct_logit_contribution)[:5]

        reconstruction_error = abs(total_contrib - baseline_logit)

        token_str = (
            self.tokenizer_decoder([target_token_id])
            if self.tokenizer_decoder is not None
            else f"Token_{target_token_id}"
        )

        return DLAResult(
            target_token_id=target_token_id,
            target_token_str=token_str,
            baseline_logit=round(baseline_logit, 4),
            embedding_contribution=round(emb_contrib, 4),
            layer_attributions=layer_attributions,
            top_promoting_heads=top_promoters,
            top_suppressing_heads=top_suppressors,
            reconstructed_logit=round(total_contrib, 4),
            reconstruction_error=round(reconstruction_error, 4),
        )

    def trace_inter_head_composition(
        self,
        layer_traces: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Traces information flow across attention heads between consecutive layers
        (Q-composition and K/V-composition).
        """
        num_layers = len(layer_traces)
        composition_circuits = []

        for l1 in range(num_layers - 1):
            l2 = l1 + 1
            w_o_l1 = self.model.blocks[l1].attn.w_o
            w_q_l2 = self.model.blocks[l2].attn.w_q
            w_k_l2 = self.model.blocks[l2].attn.w_k

            # Q-composition matrix norm: ||W_O^(l1) @ W_Q^(l2)||_F
            q_comp_norm = float(np.linalg.norm(np.matmul(w_o_l1, w_q_l2)))
            k_comp_norm = float(np.linalg.norm(np.matmul(w_o_l1, w_k_l2)))

            composition_circuits.append({
                "source_layer": l1,
                "target_layer": l2,
                "q_composition_strength": round(q_comp_norm, 4),
                "kv_composition_strength": round(k_comp_norm, 4),
                "is_active_circuit": q_comp_norm > 1.0,
            })

        return composition_circuits
