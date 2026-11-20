# Nebula

Nebula is an industry-grade mechanistic interpretability and real-time visualization suite engineered for Large Language Model inference. It demystifies the internal operational dynamics of autoregressive transformers, providing learners, researchers, and systems engineers with a granular, step-by-step microscope into how tokens flow, attention heads attend, Mixture of Experts route, residual streams evolve, and neuron activations crystallize into generated text.

---

## Executive Summary

Large Language Models are often treated as black boxes where textual prompts are submitted and generated sequences emerge without visibility into the underlying computational fabric. Nebula opens this black box during live inference. Every tokenization boundary, attention score matrix, Key-Value cache tensor update, gating decision across sparse expert networks, and unembedded thought trajectory is exposed in real time through an interactive, high-performance telemetry dashboard and programmatic inspection engine.

---

## Problem Solved

Modern generative AI practitioners face significant hurdles when attempting to develop an intuitive, rigorous understanding of internal model dynamics:

1. **Opacified Inference Mechanics**: Standard inference frameworks optimize for throughput and low latency, deliberately discarding intermediate activation vectors, attention maps, and KV cache states immediately after tensor multiplication.
2. **MoE Routing Inscrutability**: Mixture of Experts architectures select sparse sub-networks per token via learned gating routers. Without visual telemetry, observing expert specialization, routing collapse, and top-k routing entropy remains theoretical.
3. **KV Cache Memory Invisibility**: The transition between prompt prefill and sequential autoregressive decoding is a primary bottleneck in production systems. Learners struggle to visualize how Key and Value memory blocks are allocated and accessed across attention layers.
4. **Counterfactual Causality Gap**: Proving whether a specific neuron or attention head caused a specific output token requires activation patching. Without dedicated tooling, researchers cannot cleanly ablate or steer hidden states mid-inference.

Nebula eliminates these hurdles by delivering an end-to-end educational and research environment with zero required cloud dependencies, modular state hooks, and live step-by-step inspection.

---

## Key Capabilities and Teardown Modules

Nebula is organized into distinct, modular subsystems designed for teardown analysis:

- **Tokenization and Byte Teardown**: Decomposes input prompts into token sequences, UTF-8 byte arrays, hexadecimal representations, and positional coordinates.
- **Attention and KV Cache Dynamics**: Tracks multi-head query, key, and value matrices, computes head entropy, detects induction anchors, and measures cache memory footprint across prefill and decode stages.
- **Mixture of Experts Routing Engine**: Teardown of gating logits, softmax routing probabilities, top-k expert selection, load balancing histograms, and routing uncertainty metrics.
- **Neuron-Level Activation Filters**: Captures pre-activation and activated tensor states across feedforward layers, measuring sparsity, dead neurons, and activation distributions.
- **Logit Lens and Thought Progression**: Projects intermediate residual stream states directly through the unembedding matrix at every layer, exposing where the model's prediction solidifies or shifts.
- **Counterfactual Activation Patching**: Enables researchers to inject custom steering vectors, scale activations, or zero-ablate specific neurons or attention heads mid-inference to observe counterfactual causal effects.
- **Prompt Sensitivity Analyzer**: Provides side-by-side comparative analysis of how instructions, chain-of-thought framing, and keyword perturbations redirect internal decision pathways.
- **Sampling Telemetry and Filters**: Exposes how temperature scaling, top-k truncation, nucleus top-p filtering, and repetition penalties prune candidate tokens before final emission.

---

## Architectural Overview

```
                      NEBULA ARCHITECTURAL WORKFLOW

  Input Prompt String
       │
       ▼
 ┌─────────────────────────┐
 │   NebulaTokenizer       │ ──> Token IDs, Byte Breakdown, Offsets
 └───────────┬─────────────┘
             │
             ▼
 ┌─────────────────────────┐
 │ Token & Pos Embeddings  │ ──> Residual Stream: h_0
 └───────────┬─────────────┘
             │
 ┌───────────┴─────────────────────────────────────────────┐
 │ TRANSFORMER LAYER STACK (Layer 0 to Layer L-1)          │
 │                                                         │
 │  ┌─────────────────────────┐                            │
 │  │ LayerNorm / RMSNorm     │                            │
 │  └───────────┬─────────────┘                            │
 │              │                                          │
 │  ┌───────────▼─────────────┐   ┌──────────────────────┐ │
 │  │ Multi-Head Attention    │<──┤ KV Cache Tracker     │ │
 │  └───────────┬─────────────┘   └──────────────────────┘ │
 │              │  + Residual Stream Addition              │
 │  ┌───────────▼─────────────┐                            │
 │  │ LayerNorm / RMSNorm     │                            │
 │  └───────────┬─────────────┘                            │
 │              │                                          │
 │  ┌───────────▼─────────────┐   ┌──────────────────────┐ │
 │  │ Dense MLP / MoE Layer   │<──┤ MoE Router Gating    │ │
 │  │ (Top-K Expert Dispatch) │   │ (Entropy & Top-K)    │ │
 │  └───────────┬─────────────┘   └──────────────────────┘ │
 │              │  + Residual Stream Addition              │
 │              │                                          │
 │  ┌───────────▼─────────────┐   ┌──────────────────────┐ │
 │  │ Activation Patcher Hook │<──┤ Vector Injection     │ │
 │  └─────────────────────────┘   └──────────────────────┘ │
 └───────────┬─────────────────────────────────────────────┘
             │
             ▼
 ┌─────────────────────────┐     ┌──────────────────────┐
 │ Logit Lens Projection   │ ──> │ Depth Prediction Map │
 └───────────┬─────────────┘     └──────────────────────┘
             │
             ▼
 ┌─────────────────────────┐     ┌──────────────────────┐
 │ Logit Filter & Sampler  │ ──> │ Temp / Top-P / Top-K │
 └───────────┬─────────────┘     └──────────────────────┘
             │
             ▼
    Next Generated Token ──> Appended to KV Cache & Streamed
```

---

## User Interface Architecture and Telemetry Dashboard

Nebula features an interactive glassmorphic web visualization interface accessible from any browser:

```
+-----------------------------------------------------------------------------------------------+
| ✦ NEBULA | LLM Mechanistic Interpretability Suite          Model Architecture Topology       |
+------------------------------------+----------------------------------------------------------+
| CONTROLS                           | GENERATED OUTPUT STREAM                                  |
| Prompt: [User Prompt Area]         | [Token 0] [Token 1] [Token 2] ... [Emitted Token Sequence]|
| Sampling: Temperature, Top-P, Top-K |                                                          |
| Max Generation Tokens              +----------------------------------------------------------+
| [Run Inference] [Stream Live]      | TABS: [1. Pipeline] [2. Tokens] [3. KV Cache] [4. MoE]   |
|                                    |       [5. Attention] [6. Logit Lens] [7. Patching Studio] |
| STEP SCRUBBER                      +----------------------------------------------------------+
| Interactive Generation Step Scrubber| KEY-VALUE CACHE LIFECYCLE                                |
| Step Index [Prev] [Play] [Next]    | Cached Token Counter | Memory Allocation | Fill Percentage|
|                                    | Layer Memory Block Allocation & Key-Value Vector Norms   |
| INTERVENTION STUDIO                +----------------------------------------------------------+
| Target Layer & Subsystem Selector  | MIXTURE OF EXPERTS ROUTING                               |
| Intervention Type & Action Select  | Gating Router Distribution & Top-K Assigned Expert Units |
| [Execute Counterfactual Run]       | Dynamic Load Balancing & Routing Entropy Telemetry       |
+------------------------------------+----------------------------------------------------------+
```

---

## Multi-Platform Installation Guide

Nebula is written in clean, modern Python and C++ with zero operating system lock-in.

### Linux Installation

```bash
git clone https://github.com/your-username/nebula.git
cd nebula
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### macOS Installation

```bash
git clone https://github.com/your-username/nebula.git
cd nebula
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Windows Installation

```powershell
git clone https://github.com/your-username/nebula.git
cd nebula
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Getting Started and Quick Execution

### Starting the Visualization Server

Launch the local visualization server and open the browser dashboard:

```bash
nebula start --host 127.0.0.1 --port 8765
```

Navigate to `http://127.0.0.1:8765` in your browser. The dashboard will automatically connect via WebSocket and render all model states.

### Running the Terminal Teardown Demo

Run an interactive inference teardown directly within your command line:

```bash
nebula demo --prompt "Your prompt goes here"
```

### Running Backward Pass Gradient Teardown

Analyze reverse gradient flow and parameter sensitivity across layers:

```bash
nebula backward --prompt "Your prompt goes here"
```

### Running Direct Logit Attribution Teardown

Decompose winning logit contributions across attention heads and feedforward blocks:

```bash
nebula dla --prompt "Your prompt goes here"
```

---

## High-Performance C++ Bare-Metal Engine

For maximum throughput and minimal memory environments, Nebula includes a native C++ implementation located in the `cpp/` directory. Written in modern C++20 with zero external library dependencies, it delivers bare-metal execution and minimal memory overhead.

### Compiling the C++ Engine

Compile the standalone binary using GNU Make:

```bash
cd cpp
make all
```

The compiled binary will be placed at `cpp/bin/nebula-cpp`.

### Running the C++ Teardown Demo

Execute a terminal-based mechanistic teardown:

```bash
./bin/nebula-cpp demo "Your prompt goes here"
```

This displays the tokenization byte array, MoE expert assignments, logit lens predictions across depths, and exact KV cache allocation metrics.

### Launching the C++ Web Visualization Server

Start the native POSIX server that directly serves the web dashboard:

```bash
./bin/nebula-cpp start 8765
```

Navigate to `http://127.0.0.1:8765` to interact with the full web visualization suite powered by the C++ engine.

---

## Mechanistic Interpretability Core Concepts

### Tokenization Breakdown

Nebula's tokenizer provides byte-level visibility into subword boundary creation. It displays character spans, UTF-8 byte sequences, hexadecimal encodings, and special token assignments `<|bos|>`, `<|eos|>`, `<|pad|>`.

### Key-Value Cache Mechanics

During autoregressive generation, recomputing previous token keys and values is avoided by persisting them in the KV cache. Nebula tracks:
- Memory expansion from initial prefill to subsequent token decode steps.
- Per-layer tensor norms for keys and values.
- Cache fill percentage relative to maximum sequence capacity.

### Mixture of Experts Routing Dynamics

For MoE layers, Nebula calculates router gating logits and passes them through a softmax distribution. The router selects the Top-K experts, renormalizes the gating weights, and tracks routing entropy. High entropy indicates distributed uncertainty, while low entropy indicates specialized expert assignment.

### Attention Head Classification

Nebula analyzes attention weights across all heads and classifies them into behavioral archetypes:
- **Prefix Anchor Heads**: Focus primarily on the initial token.
- **Previous Token Trackers**: Focus heavily along the sub-diagonal.
- **Focused Heads**: Exhibit low attention entropy, attending to specific keywords.
- **Broad Context Aggregators**: Exhibit high attention entropy, spreading context across the prompt.

### Logit Lens and Residual Stream Unembedding

The residual stream serves as the communication highway of the transformer. By applying the unembedding matrix at each layer, Nebula surfaces the hypothetical next token at every depth stage, revealing where the model's prediction solidifies.

### Counterfactual Activation Patching

Learners can specify intervention points:
- Layer index
- Stream target: Post-Attention, Post-FFN, or Residual Stream
- Intervention action: Scaling, zero-ablation, or vector injection

Nebula executes both baseline and patched passes, outputting divergence metrics and highlighting prediction shifts.

### Response Filtering and Sampling Logit Processors

Nebula records every transformation applied to output logits prior to token sampling:
- Temperature scaling
- Top-K truncation
- Top-P nucleus filtering
- Repetition and frequency penalties
- Custom logit biases

### Backward Pass and Gradient Flow Dynamics

To reveal learning and parameter sensitivity, Nebula computes analytical gradients flowing backward through the transformer:
- Softmax cross-entropy loss gradients on target tokens.
- Residual stream gradient propagation tracking attenuation or amplification across depth.
- Identification of gradient health states categorized into healthy, vanishing, or exploding flow.
- Measurement of parameter sensitivity across attention projections and expert networks.

### Direct Logit Attribution and Circuit Tracing

Nebula implements Direct Logit Attribution by projecting each individual attention head and feedforward sub-layer directly through the unembedding matrix:
- Isolates which specific attention heads actively promote candidate tokens versus suppressing alternatives.
- Quantifies exact percentage contributions to final winning logits.
- Traces inter-head composition circuits across consecutive attention layers.

### Representation Steering and Concept Injection

Researchers can manipulate internal thought representations mid-inference:
- Extracts semantic direction vectors using mean activation differences between contrasting prompts.
- Provides real-time steering vector projection into the residual stream with continuous intensity control.
- Supports single-neuron impulse injection to study individual neuron causal influence on generated tokens.

---

## Programmatic Python API

Nebula can be used programmatically in custom Python scripts and interactive environments:

```python
import numpy as np
from nebula.config import NebulaConfig
from nebula.core.tokenizer import NebulaTokenizer
from nebula.core.model import NebulaTransformer
from nebula.interpretability.logit_lens import LogitLens
from nebula.steering.patching import ActivationPatcher, PatchIntervention

# Initialize configuration and model
config = NebulaConfig()
tokenizer = NebulaTokenizer()
model = NebulaTransformer(config.model)

# Encode prompt and run inference forward pass
tokens = tokenizer.encode("Your input prompt text")
input_ids = np.array([tokens])
logits, past_kv, trace = model.forward(input_ids)

# Inspect thought progression via Logit Lens
lens = LogitLens(model.w_unembed, tokenizer_decoder=tokenizer.decode)
predictions = lens.analyze_residual_stream(trace["residual_streams"], token_pos=-1)

for pred in predictions:
    print(pred["layer_name"], pred["top_prediction"])

# Perform counterfactual activation patching
patcher = ActivationPatcher()
patcher.add_intervention(
    PatchIntervention(layer_idx=1, target="post_attn", action="zero_ablate")
)
comparison = patcher.run_counterfactual_comparison(model, input_ids)
```

---

## License

Nebula is licensed under the Apache License, Version 2.0. You may freely use, modify, and distribute this software for educational, academic, and commercial purposes.
