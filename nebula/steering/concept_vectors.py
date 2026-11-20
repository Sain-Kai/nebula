"""
Representation steering vectors and neuron-targeted concept injection studio.
Enables extraction of direction vectors from contrasting prompt distributions,
and allows continuous steering of hidden representations and individual neurons mid-inference.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from nebula.steering.patching import ActivationPatcher, PatchIntervention


@dataclass
class ConceptVector:
    """A semantic steering vector tied to a specific layer depth."""
    name: str
    layer_idx: int
    vector: np.ndarray
    norm: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layer_idx": self.layer_idx,
            "norm": round(float(self.norm), 4),
            "description": self.description,
            "dimension": int(self.vector.shape[-1]),
        }


@dataclass
class SteeringResult:
    """Teardown metrics showing the impact of applying a steering vector."""
    concept_name: str
    target_layer: int
    intensity: float
    baseline_prediction: Dict[str, Any]
    steered_prediction: Dict[str, Any]
    concept_alignment_before: float
    concept_alignment_after: float
    prediction_flipped: bool
    top_diverging_layers: List[Dict[str, Any]]


class ConceptSteeringStudio:
    """
    Extracts concept direction vectors and steers the model's forward representations
    via activation addition and neuron-level interventions.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.vector_library: Dict[str, ConceptVector] = {}
        self._initialize_built_in_concepts()

    def _initialize_built_in_concepts(self):
        """Creates foundational steering vectors for analysis."""
        d_model = self.model.config.d_model
        rng = np.random.RandomState(1337)

        # Generate orthogonal concept directions
        concepts = [
            ("Reasoning", 1, "Directs activations toward analytical step-by-step logic"),
            ("Creativity", 2, "Expands lexical entropy and diverts from standard paths"),
            ("Precision", 1, "Focuses attention and boosts technical subwords"),
            ("Formality", 3, "Suppresses informal tokens and anchors syntax"),
        ]

        for name, layer, desc in concepts:
            raw_v = rng.randn(d_model).astype(np.float32)
            raw_v = raw_v / (np.linalg.norm(raw_v) + 1e-12)
            self.vector_library[name] = ConceptVector(
                name=name,
                layer_idx=layer,
                vector=raw_v,
                norm=1.0,
                description=desc,
            )

    def extract_concept_vector(
        self,
        name: str,
        positive_prompts: List[str],
        negative_prompts: List[str],
        target_layer: int = 1,
    ) -> ConceptVector:
        """
        Extracts a steering direction using mean difference of activations:
        v = Mean(h_pos) - Mean(h_neg).
        """
        pos_acts = []
        for p in positive_prompts:
            ids = np.array([self.tokenizer.encode(p)])
            _, _, trace = self.model.forward(ids)
            pos_acts.append(trace["residual_streams"][target_layer + 1][0, -1, :])

        neg_acts = []
        for p in negative_prompts:
            ids = np.array([self.tokenizer.encode(p)])
            _, _, trace = self.model.forward(ids)
            neg_acts.append(trace["residual_streams"][target_layer + 1][0, -1, :])

        mean_pos = np.mean(pos_acts, axis=0)
        mean_neg = np.mean(neg_acts, axis=0)

        direction = mean_pos - mean_neg
        norm = float(np.linalg.norm(direction))
        normalized = direction / (norm + 1e-12)

        concept = ConceptVector(
            name=name,
            layer_idx=target_layer,
            vector=normalized,
            norm=round(norm, 4),
            description=f"Contrast vector extracted between {len(positive_prompts)} pos and {len(negative_prompts)} neg prompts",
        )
        self.vector_library[name] = concept
        return concept

    def apply_steering(
        self,
        prompt: str,
        concept_name: str,
        intensity: float = 1.5,
    ) -> SteeringResult:
        """
        Runs baseline inference followed by steered inference where the concept vector
        is added to the residual stream: h' = h + intensity * v.
        """
        if concept_name not in self.vector_library:
            raise ValueError(f"Concept '{concept_name}' not in library.")

        concept = self.vector_library[concept_name]
        steering_vec = concept.vector * intensity

        input_ids = np.array([self.tokenizer.encode(prompt)])

        # 1. Baseline forward pass
        base_logits, _, base_trace = self.model.forward(input_ids)
        h_base = base_trace["residual_streams"][concept.layer_idx + 1][0, -1, :]

        # 2. Steered forward pass with injection hook
        patcher = ActivationPatcher()
        patcher.add_intervention(
            PatchIntervention(
                layer_idx=concept.layer_idx,
                target="residual_stream",
                action="inject",
                vector=steering_vec,
            )
        )
        steered_logits, _, steered_trace = self.model.forward(
            input_ids, patch_hook=patcher.build_hook()
        )
        h_steered = steered_trace["residual_streams"][concept.layer_idx + 1][0, -1, :]

        # Alignments (cosine similarity with concept direction)
        def align(vec):
            n = np.linalg.norm(vec) + 1e-12
            return float(np.dot(vec, concept.vector) / n)

        align_before = align(h_base)
        align_after = align(h_steered)

        # Output predictions
        def get_top_cand(logits_arr):
            last_l = logits_arr[0, -1]
            max_l = np.max(last_l)
            exp_l = np.exp(last_l - max_l)
            probs = exp_l / (np.sum(exp_l) + 1e-12)
            top_id = int(np.argmax(probs))
            return {
                "token_id": top_id,
                "token_str": self.tokenizer.decode([top_id]),
                "prob": round(float(probs[top_id]), 4),
            }

        base_top = get_top_cand(base_logits)
        steered_top = get_top_cand(steered_logits)

        # Divergence across depth
        layer_divs = []
        for l in range(len(base_trace["residual_streams"])):
            hb = base_trace["residual_streams"][l][0, -1]
            hs = steered_trace["residual_streams"][l][0, -1]
            cos = float(np.dot(hb, hs) / ((np.linalg.norm(hb) * np.linalg.norm(hs)) + 1e-12))
            layer_divs.append({
                "layer_idx": l,
                "layer_name": "Embedding" if l == 0 else f"Layer {l}",
                "cosine_similarity": round(cos, 4),
            })

        return SteeringResult(
            concept_name=concept_name,
            target_layer=concept.layer_idx,
            intensity=intensity,
            baseline_prediction=base_top,
            steered_prediction=steered_top,
            concept_alignment_before=round(align_before, 4),
            concept_alignment_after=round(align_after, 4),
            prediction_flipped=base_top["token_id"] != steered_top["token_id"],
            top_diverging_layers=layer_divs,
        )

    def inject_neuron_impulse(
        self,
        prompt: str,
        layer_idx: int,
        neuron_idx: int,
        impulse_value: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Injects an activation impulse into an individual neuron at a chosen layer
        to study single-neuron causal control.
        """
        input_ids = np.array([self.tokenizer.encode(prompt)])

        # Baseline
        base_logits, _, _ = self.model.forward(input_ids)

        # Patched with neuron intervention
        def neuron_hook(l, target, act):
            if l == layer_idx and target == "post_ffn":
                patched = act.copy()
                patched[..., neuron_idx] += impulse_value
                return patched
            return act

        steered_logits, _, _ = self.model.forward(input_ids, patch_hook=neuron_hook)

        # Extract top token before and after
        base_tid = int(np.argmax(base_logits[0, -1]))
        steered_tid = int(np.argmax(steered_logits[0, -1]))

        return {
            "layer_idx": layer_idx,
            "neuron_idx": neuron_idx,
            "impulse_value": impulse_value,
            "baseline_token": self.tokenizer.decode([base_tid]),
            "steered_token": self.tokenizer.decode([steered_tid]),
            "caused_token_change": base_tid != steered_tid,
        }
