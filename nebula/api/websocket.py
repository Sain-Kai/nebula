"""
WebSocket streaming handler for live token-by-token LLM inference teardown.
"""

from __future__ import annotations
import asyncio
import json
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from nebula.interpretability.kv_cache import KVCacheTracker
from nebula.interpretability.logit_lens import LogitLens
from nebula.interpretability.attention_analyzer import AttentionAnalyzer
from nebula.interpretability.neuron_filter import NeuronFilter
from nebula.steering.patching import ActivationPatcher, PatchIntervention
from nebula.sampling.filters import LogitFilter


async def handle_websocket_inference(websocket: WebSocket, model, tokenizer, config):
    await websocket.accept()
    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)

            prompt = data.get("prompt", "Hello Nebula")
            max_new_tokens = int(data.get("max_new_tokens", 16))
            temperature = float(data.get("temperature", 0.7))
            top_k = int(data.get("top_k", 40))
            top_p = float(data.get("top_p", 0.9))
            repetition_penalty = float(data.get("repetition_penalty", 1.05))
            stream_delay_ms = int(data.get("stream_delay_ms", 50))
            patches_data = data.get("patches", [])

            # Configure patcher
            patcher = ActivationPatcher()
            for p in patches_data:
                patcher.add_intervention(
                    PatchIntervention(
                        layer_idx=int(p["layer_idx"]),
                        target=p["target"],
                        action=p["action"],
                        neuron_idx=p.get("neuron_idx"),
                        scalar_multiplier=float(p.get("scalar_multiplier", 1.0)),
                    )
                )

            patch_hook = patcher.build_hook() if patcher.interventions else None

            # Setup tools
            kv_tracker = KVCacheTracker(
                n_layers=config.model.n_layers,
                n_heads=config.model.n_heads,
                d_head=config.model.d_model // config.model.n_heads,
                max_seq_len=config.model.max_seq_len,
            )
            logit_lens = LogitLens(model.w_unembed, tokenizer_decoder=tokenizer.decode)
            attn_analyzer = AttentionAnalyzer(n_heads=config.model.n_heads)
            neuron_filter = NeuronFilter()
            logit_filter = LogitFilter(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Tokenize prompt
            prompt_tokens = tokenizer.encode(prompt)
            current_tokens = list(prompt_tokens)

            # 1. Prefill step
            input_tensor = np.array([prompt_tokens])
            logits, past_kv, trace = model.forward(input_tensor, past_kv=None, patch_hook=patch_hook)
            kv_snapshot = kv_tracker.record_step(0, past_kv)

            # Analyze prefill trace
            last_logits = logits[0, -1]
            sampled_token, sampling_telemetry = logit_filter.process_logits(
                last_logits, generated_tokens=current_tokens
            )

            # Extract prefill layer info
            prefill_moe = []
            prefill_attention = []
            prefill_neurons = []
            for l_idx, lt in enumerate(trace["layer_traces"]):
                if lt["is_moe"]:
                    prefill_moe.append({
                        "layer_idx": l_idx,
                        "top_k_indices": lt["ffn_or_moe"]["top_k_indices"][0, -1].tolist(),
                        "top_k_weights": [round(float(w), 4) for w in lt["ffn_or_moe"]["top_k_weights"][0, -1]],
                    })
                attn_rep = attn_analyzer.analyze_layer_attention(lt["attention"]["attention_weights"])
                prefill_attention.append({
                    "layer_idx": l_idx,
                    "heads": attn_rep["head_analyses"],
                })
                # Neuron activity
                if not lt["is_moe"]:
                    n_rep = neuron_filter.analyze_layer_neurons(lt["ffn_or_moe"]["neuron_activations"], layer_idx=l_idx)
                    prefill_neurons.append(n_rep)

            lens_layers = logit_lens.analyze_residual_stream(trace["residual_streams"], token_pos=-1)

            prefill_frame = {
                "event": "step",
                "step": 0,
                "is_prefill": True,
                "token_id": prompt_tokens[-1],
                "token_str": tokenizer.decode([prompt_tokens[-1]]),
                "all_prompt_tokens": prompt_tokens,
                "kv_cache": kv_snapshot.to_dict(),
                "moe_routing": prefill_moe,
                "logit_lens": lens_layers,
                "attention_summary": prefill_attention,
                "neuron_highlights": prefill_neurons,
                "sampling_telemetry": sampling_telemetry,
                "next_token_id": sampled_token,
                "next_token_str": tokenizer.decode([sampled_token]),
            }

            await websocket.send_text(json.dumps(prefill_frame))
            if stream_delay_ms > 0:
                await asyncio.sleep(stream_delay_ms / 1000.0)

            # 2. Autoregressive decode steps
            generated_ids = []
            for step_idx in range(1, max_new_tokens + 1):
                next_in = np.array([[sampled_token]])
                current_tokens.append(sampled_token)
                generated_ids.append(sampled_token)

                logits, past_kv, trace = model.forward(next_in, past_kv=past_kv, patch_hook=patch_hook)
                kv_snapshot = kv_tracker.record_step(step_idx, past_kv, current_token_id=sampled_token)

                next_logits = logits[0, -1]
                sampled_token, sampling_telemetry = logit_filter.process_logits(
                    next_logits, generated_tokens=current_tokens
                )

                step_moe = []
                step_attn = []
                step_neurons = []
                for l_idx, lt in enumerate(trace["layer_traces"]):
                    if lt["is_moe"]:
                        step_moe.append({
                            "layer_idx": l_idx,
                            "top_k_indices": lt["ffn_or_moe"]["top_k_indices"][0, -1].tolist(),
                            "top_k_weights": [round(float(w), 4) for w in lt["ffn_or_moe"]["top_k_weights"][0, -1]],
                        })
                    attn_rep = attn_analyzer.analyze_layer_attention(lt["attention"]["attention_weights"])
                    step_attn.append({
                        "layer_idx": l_idx,
                        "heads": attn_rep["head_analyses"],
                    })
                    if not lt["is_moe"]:
                        n_rep = neuron_filter.analyze_layer_neurons(lt["ffn_or_moe"]["neuron_activations"], layer_idx=l_idx)
                        step_neurons.append(n_rep)

                lens_layers = logit_lens.analyze_residual_stream(trace["residual_streams"], token_pos=-1)

                step_frame = {
                    "event": "step",
                    "step": step_idx,
                    "is_prefill": False,
                    "token_id": current_tokens[-1],
                    "token_str": tokenizer.decode([current_tokens[-1]]),
                    "kv_cache": kv_snapshot.to_dict(),
                    "moe_routing": step_moe,
                    "logit_lens": lens_layers,
                    "attention_summary": step_attn,
                    "neuron_highlights": step_neurons,
                    "sampling_telemetry": sampling_telemetry,
                    "next_token_id": sampled_token,
                    "next_token_str": tokenizer.decode([sampled_token]),
                }

                await websocket.send_text(json.dumps(step_frame))

                # Check EOS
                if sampled_token == tokenizer.EOS_TOKEN_ID:
                    break

                if stream_delay_ms > 0:
                    await asyncio.sleep(stream_delay_ms / 1000.0)

            # Completion event
            complete_frame = {
                "event": "complete",
                "total_generated": len(generated_ids),
                "generated_text": tokenizer.decode(generated_ids),
                "full_output": tokenizer.decode(current_tokens),
            }
            await websocket.send_text(json.dumps(complete_frame))

    except WebSocketDisconnect:
        pass
