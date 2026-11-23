"""
Nebula Command-Line Interface.
"""

from __future__ import annotations
import argparse
import sys
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from nebula.config import NebulaConfig
from nebula.core.tokenizer import NebulaTokenizer
from nebula.core.model import NebulaTransformer
from nebula.interpretability.logit_lens import LogitLens
from nebula.interpretability.attention_analyzer import AttentionAnalyzer
from nebula.interpretability.circuit_tracer import CircuitTracer
from nebula.core.backward import BackwardPassEngine
from nebula.api.server import create_app

console = Console()


def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]✦ NEBULA[/bold cyan] [dim]|[/dim] [bold white]LLM Mechanistic Interpretability Suite[/bold white]\n"
            "[dim]Fine-grained layer, neuron, KV cache, and MoE inference teardown studio[/dim]",
            border_style="cyan",
        )
    )


def run_demo(prompt: str = "Nebula mechanistic interpretability"):
    print_banner()
    console.print(f"\n[bold green]Running inference teardown on prompt:[/bold green] [italic]'{prompt}'[/italic]\n")

    cfg = NebulaConfig()
    tok = NebulaTokenizer()
    model = NebulaTransformer(cfg.model, seed=42)

    # 1. Tokenization Breakdown
    tokens = tok.analyze(prompt)
    t_table = Table(title="1. Tokenization & Byte Breakdown", border_style="dim")
    t_table.add_column("Idx", justify="right", style="cyan")
    t_table.add_column("Token Text", style="bold white")
    t_table.add_column("Token ID", justify="right", style="magenta")
    t_table.add_column("Hex Bytes", style="yellow")
    t_table.add_column("Character Span", style="dim")

    for t in tokens:
        t_table.add_row(str(t.index), t.text, str(t.token_id), t.to_dict()["hex"], f"[{t.start_char}..{t.end_char}]")
    console.print(t_table)

    # 2. Forward Pass Execution
    import numpy as np
    input_ids = np.array([[t.token_id for t in tokens]])
    logits, past_kv, trace = model.forward(input_ids)

    # 3. MoE Routing Summary
    moe_table = Table(title="2. Mixture of Experts Routing", border_style="dim")
    moe_table.add_column("Layer", style="cyan")
    moe_table.add_column("Top-K Chosen Experts", style="bold green")
    moe_table.add_column("Gating Weights", style="yellow")

    for l_idx, lt in enumerate(trace["layer_traces"]):
        if lt["is_moe"]:
            chosen = lt["ffn_or_moe"]["top_k_indices"][0, -1].tolist()
            weights = [f"{w:.3f}" for w in lt["ffn_or_moe"]["top_k_weights"][0, -1]]
            moe_table.add_row(f"Layer {l_idx}", f"Experts {chosen}", f"Weights {weights}")
    console.print(moe_table)

    # 4. Logit Lens Prediction Progression
    lens = LogitLens(model.w_unembed, tokenizer_decoder=tok.decode)
    lens_preds = lens.analyze_residual_stream(trace["residual_streams"], token_pos=-1)

    lens_table = Table(title="3. Logit Lens: Thought Progression by Depth", border_style="dim")
    lens_table.add_column("Depth", style="cyan")
    lens_table.add_column("Top Candidate", style="bold green")
    lens_table.add_column("Probability", style="yellow")
    lens_table.add_column("Entropy", style="magenta")

    for lp in lens_preds:
        top = lp["top_prediction"]
        lens_table.add_row(
            lp["layer_name"],
            repr(top["token_str"]) if top else "N/A",
            f"{top['probability'] * 100:.1f}%" if top else "N/A",
            f"{lp['entropy']:.3f}",
        )
    console.print(lens_table)
    console.print("\n[bold cyan]✓ Demo teardown completed successfully.[/bold cyan]\n")


def start_server(host: str = "127.0.0.1", port: int = 8765):
    print_banner()
    console.print(f"[bold green]Starting Nebula Web Dashboard on:[/bold green] [link=http://{host}:{port}]http://{host}:{port}[/link]\n")
    cfg = NebulaConfig()
    cfg.server.host = host
    cfg.server.port = port
    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port, log_level="info")


def run_backward(prompt: str = "Explain backpropagation", target_token_id: int = 65):
    print_banner()
    console.print(f"\n[bold green]Running analytical backward pass on prompt:[/bold green] [italic]'{prompt}'[/italic]\n")

    cfg = NebulaConfig()
    tok = NebulaTokenizer()
    model = NebulaTransformer(cfg.model, seed=42)

    import numpy as np
    input_ids = np.array([tok.encode(prompt)])
    engine = BackwardPassEngine(model, tokenizer=tok)
    telemetry = engine.compute_loss_and_gradients(input_ids, target_token_id)

    b_table = Table(title="Backward Pass Gradient Flow", border_style="dim")
    b_table.add_column("Layer", justify="right", style="cyan")
    b_table.add_column("Type", style="bold white")
    b_table.add_column("Residual Grad Norm", justify="right", style="green")
    b_table.add_column("Attn Grad Norm", justify="right", style="yellow")
    b_table.add_column("FFN Grad Norm", justify="right", style="magenta")
    b_table.add_column("Ratio to Final", justify="right", style="dim")

    for lg in telemetry.layer_gradients:
        b_table.add_row(
            f"Layer {lg.layer_idx}",
            "MoE" if lg.is_moe else "Dense",
            f"{lg.residual_grad_norm:.4f}",
            f"{lg.attn_grad_norm:.4f}",
            f"{lg.ffn_or_moe_grad_norm:.4f}",
            f"{lg.gradient_ratio_to_final:.4f}",
        )
    console.print(b_table)
    console.print(f"Loss: [bold yellow]{telemetry.loss}[/bold yellow] | Gradient Health: [bold green]{telemetry.gradient_health}[/bold green]\n")


def run_dla(prompt: str = "Direct logit attribution", target_token_id: int = 65):
    print_banner()
    console.print(f"\n[bold green]Running Direct Logit Attribution on prompt:[/bold green] [italic]'{prompt}'[/italic]\n")

    cfg = NebulaConfig()
    tok = NebulaTokenizer()
    model = NebulaTransformer(cfg.model, seed=42)

    import numpy as np
    input_ids = np.array([tok.encode(prompt)])
    logits, _, trace = model.forward(input_ids)

    tracer = CircuitTracer(model, tokenizer_decoder=tok.decode)
    dla = tracer.compute_direct_logit_attribution(
        residual_streams=trace["residual_streams"],
        layer_traces=trace["layer_traces"],
        target_token_id=target_token_id,
    )

    dla_table = Table(title=f"Direct Logit Attribution for '{dla.target_token_str}'", border_style="dim")
    dla_table.add_column("Component", style="cyan")
    dla_table.add_column("Direct Logit Contribution", justify="right", style="yellow")
    dla_table.add_column("Percentage", justify="right", style="magenta")
    dla_table.add_column("Effect", style="bold")

    dla_table.add_row("Input Embedding", f"{dla.embedding_contribution:+.4f}", "-", "Baseline")
    for h in dla.top_promoting_heads[:4]:
        dla_table.add_row(f"Layer {h.layer_idx} Head {h.head_idx}", f"{h.direct_logit_contribution:+.4f}", f"{h.percentage_contribution:.1f}%", "[green]Promoting[/green]")
    for h in dla.top_suppressing_heads[:3]:
        dla_table.add_row(f"Layer {h.layer_idx} Head {h.head_idx}", f"{h.direct_logit_contribution:+.4f}", f"{h.percentage_contribution:.1f}%", "[red]Suppressing[/red]")

    console.print(dla_table)
    console.print(f"Reconstructed Logit: [bold green]{dla.reconstructed_logit:.4f}[/bold green] | Baseline: [bold yellow]{dla.baseline_logit:.4f}[/bold yellow]\n")


def run_deliberation(prompt: str = "Because reasoning matters", max_tokens: int = 12):
    print_banner()
    console.print(f"\n[bold green]Running cognitive deliberation analysis on prompt:[/bold green] [italic]'{prompt}'[/italic]\n")
    import numpy as np
    from nebula.cognitive.deliberation import CognitiveEvaluator
    cfg = NebulaConfig()
    tok = NebulaTokenizer()
    model = NebulaTransformer(cfg.model, seed=42)

    input_ids = tok.encode(prompt)
    curr_ids = list(input_ids)
    past_kv = None
    emitted = []
    logits_hist = []
    for _ in range(max_tokens):
        inp = np.array([curr_ids], dtype=np.int64)
        logits, past_kv, _ = model.forward(inp, past_kv=past_kv)
        step_logits = logits[0, -1, :]
        logits_hist.append(step_logits)
        nxt = int(np.argmax(step_logits))
        emitted.append(nxt)
        curr_ids = [nxt]
        if nxt == tok.EOS_TOKEN_ID:
            break

    evaluator = CognitiveEvaluator(tok)
    trace = evaluator.evaluate_generation_trace(emitted, logits_hist)

    delib_table = Table(title="Cognitive Deliberation Dynamics", border_style="dim")
    delib_table.add_column("Step", justify="right", style="cyan")
    delib_table.add_column("Emitted Token", style="bold white")
    delib_table.add_column("Confidence", justify="right", style="green")
    delib_table.add_column("Entropy", justify="right", style="magenta")
    delib_table.add_column("Hesitation Point", style="yellow")
    delib_table.add_column("Top Alternative", style="dim")

    for s in trace.steps:
        delib_table.add_row(
            str(s.step_idx + 1),
            repr(s.token_str),
            f"{s.probability * 100:.1f}%",
            f"{s.entropy:.3f}",
            "[red]Yes (Hesitation)[/red]" if s.is_hesitation_point else "[green]Fluent[/green]",
            f"{repr(s.top_alternative_token)} ({s.top_alternative_prob * 100:.1f}%)",
        )
    console.print(delib_table)
    console.print(f"Cognitive State: [bold cyan]{trace.cognitive_certainty}[/bold cyan] | Deliberation Ratio: [bold yellow]{trace.deliberation_ratio * 100:.1f}%[/bold yellow] | Mean Entropy: [bold magenta]{trace.mean_entropy:.3f}[/bold magenta]\n")


def run_sae(prompt: str = "Sparse Autoencoder", layer_idx: int = 1, k_sparsity: int = 16):
    print_banner()
    console.print(f"\n[bold green]Running Sparse Autoencoder decomposition at Layer {layer_idx}:[/bold green]\n")
    import numpy as np
    from nebula.interpretability.sae import SparseAutoencoder
    from nebula.interpretability.feature_catalog import FeatureCatalog
    cfg = NebulaConfig()
    tok = NebulaTokenizer()
    model = NebulaTransformer(cfg.model, seed=42)

    input_ids = tok.encode(prompt)
    inp = np.array([input_ids], dtype=np.int64)
    _, _, trace = model.forward(inp)

    res_stream = trace["residual_streams"][layer_idx + 1]
    x_vec = res_stream[0, -1, :]

    sae = SparseAutoencoder(d_model=model.config.d_model, expansion_factor=4, k_sparsity=k_sparsity, layer_idx=layer_idx)
    catalog = FeatureCatalog(tok)
    catalog.auto_tag_sae(sae)

    res = sae.decompose(x_vec, top_n=6)

    sae_table = Table(title=f"Sparse Autoencoder Active Latents (Layer {layer_idx})", border_style="dim")
    sae_table.add_column("Rank", justify="right", style="cyan")
    sae_table.add_column("Feature ID", style="bold yellow")
    sae_table.add_column("Concept Label", style="white")
    sae_table.add_column("Activation", justify="right", style="green")

    for i, f in enumerate(res.active_features):
        sae_table.add_row(str(i + 1), f"#{f.feature_idx}", f.semantic_label, f"{f.activation:.4f}")
    console.print(sae_table)
    console.print(f"L0 Sparsity: [bold cyan]{res.l0_sparsity}[/bold cyan] | Explained Variance: [bold green]{res.explained_variance * 100:.1f}%[/bold green] | Reconstruction Loss: [bold magenta]{res.reconstruction_loss:.4f}[/bold magenta]\n")


def run_speculate(prefix: str = "Speculative decoding", gamma: int = 4):
    print_banner()
    console.print(f"\n[bold green]Running speculative verification (gamma={gamma}) on prefix:[/bold green] [italic]'{prefix}'[/italic]\n")
    import numpy as np
    from nebula.core.speculative import SpeculativeVerifier
    cfg = NebulaConfig()
    tok = NebulaTokenizer()
    model = NebulaTransformer(cfg.model, seed=42)

    prefix_ids = tok.encode(prefix)
    curr_ids = list(prefix_ids)
    draft_ids = []
    draft_probs = []
    past_kv = None

    for _ in range(gamma):
        inp = np.array([curr_ids], dtype=np.int64)
        logits, past_kv, _ = model.forward(inp, past_kv=past_kv)
        step_logits = logits[0, -1, :]
        exp_l = np.exp(step_logits - np.max(step_logits))
        probs = exp_l / np.sum(exp_l)
        nxt = int(np.argmax(probs))
        draft_ids.append(nxt)
        draft_probs.append(float(probs[nxt]))
        curr_ids = [nxt]

    verifier = SpeculativeVerifier(model, tok, draft_gamma=gamma)
    trace = verifier.verify_speculative_candidates(prefix_ids, draft_ids, draft_probs)

    spec_table = Table(title="Speculative Rejection Sampling Telemetry", border_style="dim")
    spec_table.add_column("Pos", justify="right", style="cyan")
    spec_table.add_column("Draft Token", style="bold white")
    spec_table.add_column("Draft Prob (q)", justify="right", style="yellow")
    spec_table.add_column("Target Prob (p)", justify="right", style="magenta")
    spec_table.add_column("Acceptance Ratio", justify="right", style="cyan")
    spec_table.add_column("Verdict", style="bold")

    for d in trace.decisions:
        verdict = "[green]ACCEPTED[/green]" if d.is_accepted else f"[red]REJECTED[/red] (resampled {repr(d.resampled_token_str)})"
        spec_table.add_row(
            str(d.position),
            repr(d.draft_token_str),
            f"{d.draft_prob:.3f}",
            f"{d.target_prob:.3f}",
            f"{d.acceptance_prob:.3f}",
            verdict,
        )
    console.print(spec_table)
    console.print(f"Proposed: [bold white]{trace.proposed_tokens}[/bold white] | Accepted: [bold green]{trace.accepted_tokens}[/bold green] | Speedup Ratio: [bold cyan]{trace.speedup_ratio:.2f}x[/bold cyan]\n")


def main():
    parser = argparse.ArgumentParser(description="Nebula: LLM Mechanistic Interpretability Suite")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Start command
    srv_parser = subparsers.add_parser("start", help="Start the web visualization server")
    srv_parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    srv_parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive terminal teardown demo")
    demo_parser.add_argument("--prompt", default="Nebula mechanistic interpretability", help="Prompt text")

    # Backward command
    bw_parser = subparsers.add_parser("backward", help="Run analytical backward pass teardown")
    bw_parser.add_argument("--prompt", default="Explain backpropagation", help="Prompt text")
    bw_parser.add_argument("--target", type=int, default=65, help="Target token ID")

    # DLA command
    dla_parser = subparsers.add_parser("dla", help="Run direct logit attribution decomposition")
    dla_parser.add_argument("--prompt", default="Direct logit attribution", help="Prompt text")
    dla_parser.add_argument("--target", type=int, default=65, help="Target token ID")

    # Deliberate command
    delib_parser = subparsers.add_parser("deliberate", help="Evaluate cognitive reasoning and deliberation dynamics")
    delib_parser.add_argument("--prompt", default="Because reasoning matters", help="Prompt text")
    delib_parser.add_argument("--max-tokens", type=int, default=12, help="Max tokens to evaluate")

    # SAE command
    sae_parser = subparsers.add_parser("sae", help="Decompose residual activations with Sparse Autoencoders")
    sae_parser.add_argument("--prompt", default="Sparse Autoencoder", help="Prompt text")
    sae_parser.add_argument("--layer", type=int, default=1, help="Layer index")
    sae_parser.add_argument("--k", type=int, default=16, help="Top-K sparsity constraint")

    # Speculate command
    spec_parser = subparsers.add_parser("speculate", help="Verify speculative decoding multi-candidate proposals")
    spec_parser.add_argument("--prefix", default="Speculative decoding", help="Prefix text")
    spec_parser.add_argument("--gamma", type=int, default=4, help="Draft lookahead tokens")

    args = parser.parse_args()

    if args.command == "start":
        start_server(host=args.host, port=args.port)
    elif args.command == "demo":
        run_demo(prompt=args.prompt)
    elif args.command == "backward":
        run_backward(prompt=args.prompt, target_token_id=args.target)
    elif args.command == "dla":
        run_dla(prompt=args.prompt, target_token_id=args.target)
    elif args.command == "deliberate":
        run_deliberation(prompt=args.prompt, max_tokens=args.max_tokens)
    elif args.command == "sae":
        run_sae(prompt=args.prompt, layer_idx=args.layer, k_sparsity=args.k)
    elif args.command == "speculate":
        run_speculate(prefix=args.prefix, gamma=args.gamma)
    else:
        run_demo()


if __name__ == "__main__":
    main()
