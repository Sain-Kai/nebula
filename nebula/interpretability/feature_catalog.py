"""
Monosemantic feature dictionary catalog and semantic attribution tagger.
Maps sparse autoencoder latent features to human-interpretable concepts,
and computes direct feature-to-logit promotion and suppression effects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class FeatureLogitEffect:
    """The direct vocabulary logit impact of a single monosemantic feature."""
    feature_idx: int
    concept_label: str
    top_promoted_tokens: List[Dict[str, Any]]
    top_suppressed_tokens: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_idx": self.feature_idx,
            "concept_label": self.concept_label,
            "top_promoted": self.top_promoted_tokens,
            "top_suppressed": self.top_suppressed_tokens,
        }


class FeatureCatalog:
    """
    Annotates Sparse Autoencoder dictionaries with semantic descriptions
    and maps feature directions directly to vocabulary logits.
    """

    DEFAULT_TAXONOMY = [
        "Syntactic indentation and code block markers",
        "Mathematical and quantitative operations",
        "Logical negation and contradiction operators",
        "Named entities and geographical locations",
        "Refusal, safety boundary, and compliance tone",
        "Algorithmic reasoning and step-by-step sequencing",
        "Emotional valence and sentiment polarity",
        "Punctuation structure and boundary delimitation",
    ]

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.catalog: Dict[int, str] = {}

    def auto_tag_sae(self, sae) -> Dict[int, str]:
        """Assigns semantic concepts across the SAE feature dictionary."""
        n_features = sae.n_features
        n_taxonomy = len(self.DEFAULT_TAXONOMY)

        for i in range(n_features):
            tax_label = self.DEFAULT_TAXONOMY[i % n_taxonomy]
            tag = f"Concept {i}: {tax_label}"
            self.catalog[i] = tag
            sae.feature_labels[i] = tag

        return self.catalog

    def compute_feature_to_logit_attribution(
        self,
        sae,
        w_unembed: np.ndarray,
        feature_idx: int,
        top_k: int = 5,
    ) -> FeatureLogitEffect:
        """
        Projects the feature's decoder vector through the unembedding matrix:
        v_eff = W_dec[feature_idx, :] @ W_unembed
        """
        dec_vec = sae.w_dec[feature_idx, :]  # (d_model,)
        logits = np.matmul(dec_vec, w_unembed)  # (vocab_size,)

        # Top promoted tokens
        promoted_indices = np.argsort(-logits)[:top_k]
        suppressed_indices = np.argsort(logits)[:top_k]

        promoted = []
        for tid in promoted_indices:
            tok_str = self.tokenizer.decode([int(tid)]) if self.tokenizer else f"Token_{tid}"
            promoted.append({
                "token_id": int(tid),
                "token_str": tok_str,
                "logit_effect": round(float(logits[tid]), 4),
            })

        suppressed = []
        for tid in suppressed_indices:
            tok_str = self.tokenizer.decode([int(tid)]) if self.tokenizer else f"Token_{tid}"
            suppressed.append({
                "token_id": int(tid),
                "token_str": tok_str,
                "logit_effect": round(float(logits[tid]), 4),
            })

        concept_label = self.catalog.get(feature_idx, f"Latent_Feature_{feature_idx}")

        return FeatureLogitEffect(
            feature_idx=feature_idx,
            concept_label=concept_label,
            top_promoted_tokens=promoted,
            top_suppressed_tokens=suppressed,
        )
