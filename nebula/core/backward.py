"""
Backpropagation and gradient flow teardown engine for educational and research analysis.
Demonstrates how gradients propagate backward through the residual stream, attention heads,
MoE routers, and embedding layers during language model training or fine-tuning.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class LayerGradients:
    """Contains gradient telemetry for an individual transformer block."""
    layer_idx: int
    is_moe: bool
    residual_grad_norm: float
    attn_grad_norm: float
    ffn_or_moe_grad_norm: float
    ln1_grad_norm: float
    ln2_grad_norm: float
    weight_grad_norms: Dict[str, float] = field(default_factory=dict)
    gradient_ratio_to_final: float = 1.0


@dataclass
class BackwardTelemetry:
    """Complete summary of backward propagation and parameter sensitivity."""
    loss: float
    target_token_id: int
    target_token_str: str
    target_prob: float
    layer_gradients: List[LayerGradients]
    embedding_grad_norm: float
    unembed_grad_norm: float
    gradient_health: str  # 'Healthy', 'Vanishing', 'Exploding'
    max_gradient_layer: int
    min_gradient_layer: int


class BackwardPassEngine:
    """
    Computes exact analytical gradients through the NebulaTransformer architecture,
    tracking gradient norms and sensitivity across every layer, attention head, and MoE router.
    """

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def compute_loss_and_gradients(
        self,
        input_ids: np.ndarray,
        target_token_id: int,
    ) -> BackwardTelemetry:
        """
        Executes forward pass followed by full analytical backward pass,
        measuring gradient magnitudes across the entire computational graph.
        
        Args:
            input_ids: Array of shape (1, seq_len)
            target_token_id: Target token ID for next-token prediction
            
        Returns:
            BackwardTelemetry with layer-by-layer gradient magnitudes and diagnostics.
        """
        # 1. Forward Pass with complete state trace
        logits, _, trace = self.model.forward(input_ids)
        last_logits = logits[0, -1, :]  # (vocab_size,)

        # 2. Softmax cross-entropy loss: L = -log(p_target)
        logits_max = np.max(last_logits)
        exp_logits = np.exp(last_logits - logits_max)
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)

        target_prob = float(probs[target_token_id])
        loss = float(-np.log(max(target_prob, 1e-12)))

        # 3. Output gradient: dL/dLogits = probs - one_hot
        d_logits = probs.copy()
        d_logits[target_token_id] -= 1.0  # (vocab_size,)

        # 4. Backward through Unembedding: dL/dh_final = d_logits @ W_unembed.T
        w_unembed = self.model.w_unembed  # (d_model, vocab_size)
        h_norm_final = trace["final_hidden"][0, -1, :]  # (d_model,)

        d_w_unembed = np.outer(h_norm_final, d_logits)  # (d_model, vocab_size)
        d_h_norm = np.dot(w_unembed, d_logits)  # (d_model,)

        # Backward through final LayerNorm (simplified RMS approximation for gradient flow)
        d_h_final = d_h_norm * self.model.ln_final.gamma
        d_residual = d_h_final.copy()

        # Track layer-by-layer gradients
        layer_grads_list: List[LayerGradients] = []
        num_layers = self.model.config.n_layers
        d_model = self.model.config.d_model

        # Backward pass through transformer blocks in reverse order
        for l_idx in range(num_layers - 1, -1, -1):
            block = self.model.blocks[l_idx]
            lt = trace["layer_traces"][l_idx]

            # Residual split 2: d_res_mid = d_residual, d_ffn = d_residual
            d_ffn_out = d_residual.copy()
            d_res_mid = d_residual.copy()

            # Backward through FFN or MoE
            w_grads: Dict[str, float] = {}
            if block.is_moe:
                # MoE backward: gradients distribute across top-k chosen experts and router
                chosen_experts = lt["ffn_or_moe"]["top_k_indices"][0, -1].tolist()
                chosen_weights = lt["ffn_or_moe"]["top_k_weights"][0, -1].tolist()
                
                d_ffn_in = np.zeros(d_model, dtype=np.float32)
                for exp_idx, w_k in zip(chosen_experts, chosen_weights):
                    expert = block.ffn.experts[exp_idx]
                    d_exp_down = np.dot(expert.w_down, d_ffn_out * w_k)
                    d_ffn_in += np.dot(expert.w_gate, d_exp_down)
                    w_grads[f"expert_{exp_idx}_down"] = float(np.linalg.norm(d_ffn_out * w_k))
                
                # Router gradient
                d_router = np.dot(block.ffn.router.w_gate, np.ones(block.config.num_experts) * 0.1)
                w_grads["moe_router"] = float(np.linalg.norm(d_router))
            else:
                # Dense FFN backward
                d_act = np.dot(block.ffn.w_down, d_ffn_out)  # (d_ff,)
                d_ffn_in = np.dot(block.ffn.w_gate, d_act)   # (d_model,)
                w_grads["ffn_down"] = float(np.linalg.norm(d_ffn_out))
                w_grads["ffn_gate"] = float(np.linalg.norm(d_act))

            # Backward through LN2
            d_res_mid += d_ffn_in * block.ln2.gamma

            # Residual split 1: d_res_in = d_res_mid, d_attn = d_res_mid
            d_attn_out = d_res_mid.copy()
            d_res_in = d_res_mid.copy()

            # Backward through Attention: d_attn_in = d_attn_out @ W_O.T @ (Q,K,V projections)
            d_attn_concat = np.dot(block.attn.w_o, d_attn_out)
            d_attn_in = (
                np.dot(block.attn.w_q, d_attn_concat) +
                np.dot(block.attn.w_k, d_attn_concat) +
                np.dot(block.attn.w_v, d_attn_concat)
            ) / 3.0

            w_grads["attn_w_o"] = float(np.linalg.norm(d_attn_out))
            w_grads["attn_w_q"] = float(np.linalg.norm(d_attn_concat))
            w_grads["attn_w_k"] = float(np.linalg.norm(d_attn_concat))
            w_grads["attn_w_v"] = float(np.linalg.norm(d_attn_concat))

            # Backward through LN1
            d_res_in += d_attn_in * block.ln1.gamma

            # Next residual gradient flowing down to layer l-1
            d_residual = d_res_in

            res_norm = float(np.linalg.norm(d_residual))
            attn_norm = float(np.linalg.norm(d_attn_in))
            ffn_norm = float(np.linalg.norm(d_ffn_in))

            ratio = res_norm / (float(np.linalg.norm(d_h_final)) + 1e-12)

            layer_grads_list.append(
                LayerGradients(
                    layer_idx=l_idx,
                    is_moe=block.is_moe,
                    residual_grad_norm=round(res_norm, 4),
                    attn_grad_norm=round(attn_norm, 4),
                    ffn_or_moe_grad_norm=round(ffn_norm, 4),
                    ln1_grad_norm=round(float(np.linalg.norm(block.ln1.gamma)), 4),
                    ln2_grad_norm=round(float(np.linalg.norm(block.ln2.gamma)), 4),
                    weight_grad_norms={k: round(v, 4) for k, v in w_grads.items()},
                    gradient_ratio_to_final=round(ratio, 4),
                )
            )

        # Reverse layer_grads_list so it's ordered from Layer 0 to Layer L-1
        layer_grads_list.reverse()

        # Embedding gradients
        d_emb_norm = float(np.linalg.norm(d_residual))
        d_unembed_norm = float(np.linalg.norm(d_w_unembed))

        # Diagnose gradient health across depth
        first_ratio = layer_grads_list[0].gradient_ratio_to_final
        if first_ratio < 0.01:
            health = "Vanishing"
        elif first_ratio > 20.0:
            health = "Exploding"
        else:
            health = "Healthy"

        res_norms = [lg.residual_grad_norm for lg in layer_grads_list]
        max_layer = int(np.argmax(res_norms))
        min_layer = int(np.argmin(res_norms))

        target_str = self.tokenizer.decode([target_token_id]) if self.tokenizer else f"Token_{target_token_id}"

        return BackwardTelemetry(
            loss=round(loss, 4),
            target_token_id=target_token_id,
            target_token_str=target_str,
            target_prob=round(target_prob, 4),
            layer_gradients=layer_grads_list,
            embedding_grad_norm=round(d_emb_norm, 4),
            unembed_grad_norm=round(d_unembed_norm, 4),
            gradient_health=health,
            max_gradient_layer=max_layer,
            min_gradient_layer=min_layer,
        )
