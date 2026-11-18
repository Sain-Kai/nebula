"""
Pydantic data schemas for Nebula API and WebSocket telemetry.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TokenizeRequest(BaseModel):
    text: str
    add_bos: bool = True


class TokenMetadataSchema(BaseModel):
    index: int
    token_id: int
    text: str
    byte_values: List[int]
    hex: str
    is_special: bool
    start_char: int
    end_char: int


class TokenizeResponse(BaseModel):
    tokens: List[TokenMetadataSchema]
    total_tokens: int
    encoded_ids: List[int]


class PatchSpecSchema(BaseModel):
    layer_idx: int
    target: str = Field(..., description="'residual_stream', 'post_attn', 'post_ffn', 'neuron'")
    action: str = Field(..., description="'zero_ablate', 'scale', 'inject'")
    neuron_idx: Optional[int] = None
    scalar_multiplier: float = 1.0


class InferenceRequest(BaseModel):
    prompt: str = "Why is the sky blue?"
    max_new_tokens: int = 16
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    patches: Optional[List[PatchSpecSchema]] = None
    seed: Optional[int] = 42


class PromptComparisonRequest(BaseModel):
    prompt_a: str = "Explain quantum computing simply."
    prompt_b: str = "Think step by step and explain quantum computing simply."
    label_a: str = "Direct Prompt"
    label_b: str = "Chain of Thought Prompt"


class StepFrame(BaseModel):
    step: int
    token_id: int
    token_str: str
    is_prefill: bool
    kv_cache: Dict[str, Any]
    moe_routing: List[Dict[str, Any]]
    logit_lens: List[Dict[str, Any]]
    neuron_highlights: List[Dict[str, Any]]
    attention_summary: List[Dict[str, Any]]
    sampling_telemetry: Dict[str, Any]


class InferenceResponse(BaseModel):
    prompt: str
    generated_text: str
    token_ids: List[int]
    total_steps: int
    frames: List[Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    is_moe: bool
    num_experts: int
    top_k_experts: int
    moe_layers: List[int]
    activation_fn: str
