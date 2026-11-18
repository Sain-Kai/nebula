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

    args = parser.parse_args()

    if args.command == "start":
        start_server(host=args.host, port=args.port)
    elif args.command == "demo":
        run_demo(prompt=args.prompt)
    else:
        # Default to demo when run without arguments
        run_demo()


if __name__ == "__main__":
    main()
