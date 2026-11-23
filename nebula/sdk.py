"""
Nebula Interactive Python SDK and Jupyter Notebook widget bridge.
Enables one-line mechanistic interpretability teardowns directly inside Python scripts,
IPython shells, and Jupyter notebooks with native HTML visualization rendering.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np

from nebula.config import NebulaConfig
from nebula.core.tokenizer import NebulaTokenizer
from nebula.core.model import NebulaTransformer
from nebula.interpretability.logit_lens import LogitLens
from nebula.interpretability.circuit_tracer import CircuitTracer
from nebula.interpretability.sae import SparseAutoencoder
from nebula.interpretability.feature_catalog import FeatureCatalog


class InteractiveHTMLReport:
    """Wrapper that renders rich visual HTML in Jupyter and prints cleanly in standard consoles."""

    def __init__(self, title: str, html_body: str, text_fallback: str):
        self.title = title
        self.html_body = html_body
        self.text_fallback = text_fallback

    def _repr_html_(self) -> str:
        """Invoked automatically by Jupyter/Colab to render inline HTML."""
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f1522; color: #f8fafc; padding: 16px; border-radius: 8px; border: 1px solid #222e44;">
            <div style="font-weight: 800; font-size: 1rem; color: #06b6d4; margin-bottom: 8px;">✦ NEBULA | {self.title}</div>
            {self.html_body}
        </div>
        """

    def __repr__(self) -> str:
        return f"NebulaReport({self.title})\n{self.text_fallback}"


class NebulaSession:
    """Interactive session managing models, tokenizer, and interpretability pipelines."""

    def __init__(self, config: Optional[NebulaConfig] = None):
        self.config = config or NebulaConfig()
        self.tokenizer = NebulaTokenizer(vocab_size=self.config.model.vocab_size)
        self.model = NebulaTransformer(self.config.model, seed=42)
        self.logit_lens = LogitLens(self.model.w_unembed, tokenizer_decoder=self.tokenizer.decode)
        self.circuit_tracer = CircuitTracer(self.model, tokenizer_decoder=self.tokenizer.decode)
        self.sae_layer1 = SparseAutoencoder(d_model=self.config.model.d_model, layer_idx=1)
        self.catalog = FeatureCatalog(self.tokenizer)
        self.catalog.auto_tag_sae(self.sae_layer1)

    def teardown(self, prompt: str) -> InteractiveHTMLReport:
        """Executes forward pass and returns an interactive teardown summary."""
        input_ids = np.array([self.tokenizer.encode(prompt)])
        logits, _, trace = self.model.forward(input_ids)

        preds = self.logit_lens.analyze_residual_stream(trace["residual_streams"])
        top_pred = preds[-1]["top_prediction"]
        top_str = top_pred["token_str"] if top_pred else "N/A"
        top_p = top_pred["probability"] if top_pred else 0.0

        html = f"""
        <div style="margin-bottom: 8px;"><strong>Input Prompt:</strong> <em>"{prompt}"</em></div>
        <div style="margin-bottom: 12px;"><strong>Top Predicted Token:</strong> <span style="color: #10b981; font-weight: 700;">'{top_str}'</span> ({top_p * 100:.1f}%)</div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; font-family: monospace;">
            <thead><tr style="border-bottom: 1px solid #222e44; text-align: left; color: #94a3b8;"><th>Depth</th><th>Predicted Token</th><th>Probability</th><th>Entropy</th></tr></thead>
            <tbody>
        """
        for p in preds:
            top = p["top_prediction"]
            tok_text = top["token_str"] if top else "N/A"
            prob = top["probability"] if top else 0.0
            html += f"<tr><td style='padding: 6px 0; color: #06b6d4;'>{p['layer_name']}</td><td>'{tok_text}'</td><td>{prob*100:.1f}%</td><td style='color: #64748b;'>{p['entropy']:.3f}</td></tr>"

        html += "</tbody></table>"
        text_fb = f"Prompt: {prompt}\nNext Predicted Token: {top_str} ({top_p*100:.1f}%)\nLayers: {len(preds)}"
        return InteractiveHTMLReport("Inference Teardown", html, text_fb)

    def dla(self, prompt: str, target_token_id: Optional[int] = None) -> InteractiveHTMLReport:
        """Computes Direct Logit Attribution for the prompt."""
        input_ids = np.array([self.tokenizer.encode(prompt)])
        logits, _, trace = self.model.forward(input_ids)

        target_id = target_token_id or int(np.argmax(logits[0, -1, :]))
        dla_res = self.circuit_tracer.compute_direct_logit_attribution(
            trace["residual_streams"], trace["layer_traces"], target_id
        )

        html = f"""
        <div style="margin-bottom: 8px;">Target Token: <strong>'{dla_res.target_token_str}'</strong> | Baseline Logit: {dla_res.baseline_logit:.3f}</div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        """
        for h in dla_res.top_promoting_heads[:4]:
            html += f"<span style='background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #10b981; padding: 2px 6px; border-radius: 4px; font-family: monospace;'>L{h.layer_idx}H{h.head_idx}: +{h.direct_logit_contribution:.2f}</span>"
        for h in dla_res.top_suppressing_heads[:3]:
            html += f"<span style='background: rgba(244, 63, 94, 0.2); border: 1px solid #f43f5e; color: #f43f5e; padding: 2px 6px; border-radius: 4px; font-family: monospace;'>L{h.layer_idx}H{h.head_idx}: {h.direct_logit_contribution:.2f}</span>"

        html += "</div>"
        text_fb = f"DLA for '{dla_res.target_token_str}': Reconstructed Logit {dla_res.reconstructed_logit:.3f}"
        return InteractiveHTMLReport("Direct Logit Attribution", html, text_fb)


_default_session: Optional[NebulaSession] = None


def teardown(prompt: str) -> InteractiveHTMLReport:
    """One-line convenience function for instant teardown analysis."""
    global _default_session
    if _default_session is None:
        _default_session = NebulaSession()
    return _default_session.teardown(prompt)
