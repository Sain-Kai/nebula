"""
FastAPI application serving mechanistic interpretability endpoints and static UI dashboard.
"""

from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np

from nebula.config import NebulaConfig
from nebula.core.tokenizer import NebulaTokenizer
from nebula.core.model import NebulaTransformer
from nebula.api.schemas import (
    TokenizeRequest,
    TokenizeResponse,
    TokenMetadataSchema,
    InferenceRequest,
    InferenceResponse,
    PromptComparisonRequest,
    ModelInfoResponse,
)
from nebula.api.websocket import handle_websocket_inference
from nebula.steering.patching import ActivationPatcher, PatchIntervention
from nebula.steering.prompt_impact import PromptImpactAnalyzer
from nebula.interpretability.kv_cache import KVCacheTracker
from nebula.interpretability.logit_lens import LogitLens
from nebula.interpretability.attention_analyzer import AttentionAnalyzer
from nebula.interpretability.neuron_filter import NeuronFilter
from nebula.sampling.filters import LogitFilter


def create_app(config: Optional[NebulaConfig] = None) -> FastAPI:
    if config is None:
        config = NebulaConfig()

    app = FastAPI(
        title="Nebula Interpretability Suite",
        description="Real-time mechanistic interpretability and LLM inference visualization suite",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize shared model and tokenizer
    tokenizer = NebulaTokenizer(vocab_size=config.model.vocab_size)
    model = NebulaTransformer(config.model, seed=42)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/model/info", response_model=ModelInfoResponse)
    def model_info():
        return ModelInfoResponse(
            model_name="Nebula-Transformer-MoE",
            vocab_size=config.model.vocab_size,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            is_moe=config.model.is_moe,
            num_experts=config.model.num_experts,
            top_k_experts=config.model.top_k_experts,
            moe_layers=config.model.moe_layers,
            activation_fn=config.model.activation_fn,
        )

    @app.post("/api/tokenize", response_model=TokenizeResponse)
    def tokenize_endpoint(req: TokenizeRequest):
        meta_list = tokenizer.analyze(req.text, add_bos=req.add_bos)
        ids = tokenizer.encode(req.text, add_bos=req.add_bos)
        return TokenizeResponse(
            tokens=[TokenMetadataSchema(**m.to_dict()) for m in meta_list],
            total_tokens=len(ids),
            encoded_ids=ids,
        )

    @app.post("/api/inference")
    def inference_endpoint(req: InferenceRequest):
        patcher = ActivationPatcher()
        if req.patches:
            for p in req.patches:
                patcher.add_intervention(
                    PatchIntervention(
                        layer_idx=p.layer_idx,
                        target=p.target,
                        action=p.action,
                        neuron_idx=p.neuron_idx,
                        scalar_multiplier=p.scalar_multiplier,
                    )
                )

        patch_hook = patcher.build_hook() if patcher.interventions else None

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
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )

        prompt_tokens = tokenizer.encode(req.prompt)
        current_tokens = list(prompt_tokens)
        frames: List[Dict[str, Any]] = []

        # Prefill
        input_tensor = np.array([prompt_tokens])
        logits, past_kv, trace = model.forward(input_tensor, past_kv=None, patch_hook=patch_hook)
        kv_snapshot = kv_tracker.record_step(0, past_kv)

        last_logits = logits[0, -1]
        sampled_token, sampling_telemetry = logit_filter.process_logits(
            last_logits, generated_tokens=current_tokens
        )

        prefill_moe = []
        prefill_attn = []
        prefill_neurons = []
        for l_idx, lt in enumerate(trace["layer_traces"]):
            if lt["is_moe"]:
                prefill_moe.append({
                    "layer_idx": l_idx,
                    "top_k_indices": lt["ffn_or_moe"]["top_k_indices"][0, -1].tolist(),
                    "top_k_weights": [round(float(w), 4) for w in lt["ffn_or_moe"]["top_k_weights"][0, -1]],
                })
            attn_rep = attn_analyzer.analyze_layer_attention(lt["attention"]["attention_weights"])
            prefill_attn.append({
                "layer_idx": l_idx,
                "heads": attn_rep["head_analyses"],
            })
            if not lt["is_moe"]:
                n_rep = neuron_filter.analyze_layer_neurons(lt["ffn_or_moe"]["neuron_activations"], layer_idx=l_idx)
                prefill_neurons.append(n_rep)

        lens_layers = logit_lens.analyze_residual_stream(trace["residual_streams"], token_pos=-1)

        frames.append({
            "step": 0,
            "is_prefill": True,
            "token_id": prompt_tokens[-1],
            "token_str": tokenizer.decode([prompt_tokens[-1]]),
            "kv_cache": kv_snapshot.to_dict(),
            "moe_routing": prefill_moe,
            "logit_lens": lens_layers,
            "attention_summary": prefill_attn,
            "neuron_highlights": prefill_neurons,
            "sampling_telemetry": sampling_telemetry,
            "next_token_id": sampled_token,
            "next_token_str": tokenizer.decode([sampled_token]),
        })

        # Autoregressive generation
        generated_ids = []
        for step_idx in range(1, req.max_new_tokens + 1):
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

            frames.append({
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
            })

            if sampled_token == tokenizer.EOS_TOKEN_ID:
                break

        return {
            "prompt": req.prompt,
            "generated_text": tokenizer.decode(generated_ids),
            "token_ids": current_tokens,
            "total_steps": len(frames),
            "frames": frames,
        }

    @app.post("/api/patch-compare")
    def patch_compare(req: InferenceRequest):
        patcher = ActivationPatcher()
        if req.patches:
            for p in req.patches:
                patcher.add_intervention(
                    PatchIntervention(
                        layer_idx=p.layer_idx,
                        target=p.target,
                        action=p.action,
                        neuron_idx=p.neuron_idx,
                        scalar_multiplier=p.scalar_multiplier,
                    )
                )

        input_ids = np.array([tokenizer.encode(req.prompt)])
        comparison = patcher.run_counterfactual_comparison(
            model=model,
            input_ids=input_ids,
            tokenizer_decoder=tokenizer.decode,
        )
        return comparison

    @app.post("/api/prompt-impact")
    def prompt_impact_endpoint(req: PromptComparisonRequest):
        analyzer = PromptImpactAnalyzer(model, tokenizer)
        return analyzer.compare_prompts(
            prompt_a=req.prompt_a,
            prompt_b=req.prompt_b,
            label_a=req.label_a,
            label_b=req.label_b,
        )

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        await handle_websocket_inference(websocket, model, tokenizer, config)

    # Static assets serving
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    if os.path.exists(web_dir):
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/")
        def serve_index():
            index_path = os.path.join(web_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "Nebula UI assets not initialized yet."}

    return app
