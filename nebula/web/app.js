/**
 * Nebula Interactive Frontend Controller
 */

// Application State
const state = {
    frames: [],
    currentStepIndex: 0,
    isPlaying: false,
    playInterval: null,
    generatedText: '',
    tokenIds: [],
    modelInfo: null,
};

// DOM Elements
const el = {
    promptInput: document.getElementById('prompt-input'),
    paramTemp: document.getElementById('param-temp'),
    paramTopP: document.getElementById('param-topp'),
    paramTopK: document.getElementById('param-topk'),
    paramMaxTok: document.getElementById('param-maxtok'),
    valTemp: document.getElementById('val-temp'),
    valTopP: document.getElementById('val-topp'),
    valTopK: document.getElementById('val-topk'),
    valMaxTok: document.getElementById('val-maxtok'),
    btnRun: document.getElementById('btn-run'),
    btnStream: document.getElementById('btn-stream'),
    stepSlider: document.getElementById('step-slider'),
    currentStepDisplay: document.getElementById('current-step-display'),
    btnPrevStep: document.getElementById('btn-prev-step'),
    btnNextStep: document.getElementById('btn-next-step'),
    btnPlayPause: document.getElementById('btn-play-pause'),
    tokensFlow: document.getElementById('output-tokens-flow'),
    pipelineGraph: document.getElementById('pipeline-graph'),
    tokensTable: document.getElementById('tokens-breakdown-container'),
    kvMetrics: document.getElementById('kv-metrics'),
    kvMatrixView: document.getElementById('kv-matrix-view'),
    moeViewContainer: document.getElementById('moe-view-container'),
    attentionHeadsContainer: document.getElementById('attention-heads-container'),
    lensTimelineContainer: document.getElementById('lens-timeline-container'),
    neuronExplorerContainer: document.getElementById('neuron-explorer-container'),
    btnApplyPatch: document.getElementById('btn-apply-patch'),
    patchDiffContainer: document.getElementById('patch-diff-container'),
    btnComparePrompts: document.getElementById('btn-compare-prompts'),
    promptDiffContainer: document.getElementById('prompt-diff-container'),
    diffPromptA: document.getElementById('diff-prompt-a'),
    diffPromptB: document.getElementById('diff-prompt-b'),
    navTabs: document.querySelectorAll('.nav-tab'),
    tabPanes: document.querySelectorAll('.tab-pane'),
};

// Initialize listeners
function init() {
    setupTabSwitching();
    setupParamBindings();
    setupInferenceActions();
    setupScrubberControls();
    setupPatchingStudio();
    setupPromptDiffStudio();
    setupBackwardStudio();
    setupDLAStudio();
    setupSteeringStudio();
    fetchModelInfo();
}

// Tab Switching
function setupTabSwitching() {
    el.navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            el.navTabs.forEach(t => t.classList.remove('active'));
            el.tabPanes.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const targetPane = document.getElementById(tab.getAttribute('data-tab'));
            if (targetPane) targetPane.classList.add('active');
        });
    });
}

// Slider parameter displays
function setupParamBindings() {
    const bind = (slider, display) => {
        slider.addEventListener('input', () => {
            display.textContent = slider.value;
        });
    };
    bind(el.paramTemp, el.valTemp);
    bind(el.paramTopP, el.valTopP);
    bind(el.paramTopK, el.valTopK);
    bind(el.paramMaxTok, el.valMaxTok);
}

// Fetch Model metadata
async function fetchModelInfo() {
    try {
        const res = await fetch('/api/model/info');
        if (res.ok) {
            state.modelInfo = await res.json();
            const badge = document.getElementById('model-name-text');
            if (badge && state.modelInfo) {
                badge.textContent = `${state.modelInfo.model_name} (${state.modelInfo.n_layers}L / ${state.modelInfo.n_heads}H / ${state.modelInfo.num_experts}E)`;
            }
        }
    } catch (e) {
        console.warn('Backend not responding yet:', e);
    }
}

// Inference Execution
function setupInferenceActions() {
    el.btnRun.addEventListener('click', runBatchInference);
    el.btnStream.addEventListener('click', runStreamInference);
}

async function runBatchInference() {
    el.btnRun.disabled = true;
    el.btnRun.innerHTML = '<span>⏳</span> Processing...';

    const payload = {
        prompt: el.promptInput.value,
        max_new_tokens: parseInt(el.paramMaxTok.value),
        temperature: parseFloat(el.paramTemp.value),
        top_k: parseInt(el.paramTopK.value),
        top_p: parseFloat(el.paramTopP.value),
    };

    try {
        const res = await fetch('/api/inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error('Inference request failed');
        const data = await res.json();

        state.frames = data.frames || [];
        state.generatedText = data.generated_text;
        state.tokenIds = data.token_ids;

        // Also fetch token breakdown table
        fetchTokenTeardown(payload.prompt);

        setupTimelineFrames();
        renderStep(state.frames.length - 1);
    } catch (err) {
        alert('Error executing inference: ' + err.message);
    } finally {
        el.btnRun.disabled = false;
        el.btnRun.innerHTML = '<span class="btn-icon">▶</span> Run Inference';
    }
}

function runStreamInference() {
    el.btnStream.disabled = true;
    el.btnStream.innerHTML = '<span>⚡</span> Streaming...';

    state.frames = [];
    el.tokensFlow.innerHTML = '';

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        const msg = {
            prompt: el.promptInput.value,
            max_new_tokens: parseInt(el.paramMaxTok.value),
            temperature: parseFloat(el.paramTemp.value),
            top_k: parseInt(el.paramTopK.value),
            top_p: parseFloat(el.paramTopP.value),
            stream_delay_ms: 60,
        };
        ws.send(JSON.stringify(msg));
    };

    ws.onmessage = (event) => {
        const frame = JSON.parse(event.data);
        if (frame.event === 'step') {
            state.frames.push(frame);
            setupTimelineFrames();
            renderStep(state.frames.length - 1);
        } else if (frame.event === 'complete') {
            el.btnStream.disabled = false;
            el.btnStream.innerHTML = '<span class="btn-icon">⚡</span> Stream Step-by-Step';
            ws.close();
        }
    };

    ws.onerror = () => {
        el.btnStream.disabled = false;
        el.btnStream.innerHTML = '<span class="btn-icon">⚡</span> Stream Step-by-Step';
    };
}

async function fetchTokenTeardown(text) {
    try {
        const res = await fetch('/api/tokenize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        if (res.ok) {
            const data = await res.json();
            renderTokensTable(data.tokens);
        }
    } catch (e) {
        console.error(e);
    }
}

// Scrubber setup
function setupTimelineFrames() {
    const total = state.frames.length;
    if (total === 0) return;

    el.stepSlider.disabled = false;
    el.stepSlider.min = 0;
    el.stepSlider.max = total - 1;
    el.stepSlider.value = total - 1;

    el.btnPrevStep.disabled = false;
    el.btnNextStep.disabled = false;
    el.btnPlayPause.disabled = false;

    renderTokensBanner();
}

function setupScrubberControls() {
    el.stepSlider.addEventListener('input', (e) => {
        renderStep(parseInt(e.target.value));
    });

    el.btnPrevStep.addEventListener('click', () => {
        if (state.currentStepIndex > 0) {
            renderStep(state.currentStepIndex - 1);
        }
    });

    el.btnNextStep.addEventListener('click', () => {
        if (state.currentStepIndex < state.frames.length - 1) {
            renderStep(state.currentStepIndex + 1);
        }
    });

    el.btnPlayPause.addEventListener('click', togglePlayPause);
}

function togglePlayPause() {
    if (state.isPlaying) {
        clearInterval(state.playInterval);
        state.isPlaying = false;
        el.btnPlayPause.textContent = 'Play';
    } else {
        state.isPlaying = true;
        el.btnPlayPause.textContent = 'Pause';
        state.playInterval = setInterval(() => {
            if (state.currentStepIndex < state.frames.length - 1) {
                renderStep(state.currentStepIndex + 1);
            } else {
                renderStep(0);
            }
        }, 400);
    }
}

// Render Step Details
function renderStep(stepIdx) {
    if (!state.frames[stepIdx]) return;
    state.currentStepIndex = stepIdx;
    el.stepSlider.value = stepIdx;
    const frame = state.frames[stepIdx];

    el.currentStepDisplay.textContent = `Step ${stepIdx} / ${state.frames.length - 1} (${frame.is_prefill ? 'Prefill' : 'Decode'})`;

    // Highlight token in banner
    updateBannerHighlight(stepIdx);

    // Render components
    renderPipelineOverview(frame);
    renderKVCache(frame.kv_cache);
    renderMoE(frame.moe_routing);
    renderAttention(frame.attention_summary);
    renderLogitLens(frame.logit_lens);
    renderNeurons(frame.neuron_highlights);
}

// Render Output Banner
function renderTokensBanner() {
    el.tokensFlow.innerHTML = '';
    state.frames.forEach((f, idx) => {
        const span = document.createElement('span');
        span.className = `token-pill ${f.is_prefill ? 'prompt-token' : 'generated-token'}`;
        span.textContent = f.token_str || `T_${f.token_id}`;
        span.title = `Token ID: ${f.token_id} | Step: ${idx}`;
        span.addEventListener('click', () => renderStep(idx));
        el.tokensFlow.appendChild(span);
    });
}

function updateBannerHighlight(stepIdx) {
    const pills = el.tokensFlow.querySelectorAll('.token-pill');
    pills.forEach((p, idx) => {
        if (idx === stepIdx) {
            p.classList.add('active');
        } else {
            p.classList.remove('active');
        }
    });
}

// Render Pipeline Overview
function renderPipelineOverview(frame) {
    el.pipelineGraph.innerHTML = `
        <div class="pipeline-flow-cards" style="display: flex; gap: 12px; align-items: center; overflow-x: auto; padding: 12px 0;">
            <div class="metric-card" style="min-width: 140px;">
                <span class="metric-label">INPUT TOKEN</span>
                <span class="metric-value" style="font-size: 1rem; color: #a855f7;">${frame.token_str}</span>
                <span style="font-size: 0.75rem; color: #64748b;">ID: ${frame.token_id}</span>
            </div>
            <span style="color: #64748b;">➔</span>
            <div class="metric-card" style="min-width: 140px;">
                <span class="metric-label">KV CACHE STATE</span>
                <span class="metric-value" style="font-size: 1rem;">${frame.kv_cache.total_cached_tokens} Tokens</span>
                <span style="font-size: 0.75rem; color: #06b6d4;">${frame.kv_cache.memory_kb} KB</span>
            </div>
            <span style="color: #64748b;">➔</span>
            <div class="metric-card" style="min-width: 140px;">
                <span class="metric-label">MOE EXPERTS</span>
                <span class="metric-value" style="font-size: 1rem; color: #10b981;">${frame.moe_routing ? frame.moe_routing.length : 0} MoE Layers</span>
                <span style="font-size: 0.75rem; color: #64748b;">Gated Top-2</span>
            </div>
            <span style="color: #64748b;">➔</span>
            <div class="metric-card" style="min-width: 140px;">
                <span class="metric-label">SAMPLING PREDICTION</span>
                <span class="metric-value" style="font-size: 1rem; color: #f59e0b;">${frame.next_token_str}</span>
                <span style="font-size: 0.75rem; color: #64748b;">Next Emitted</span>
            </div>
        </div>
    `;
}

// Render Tokenizer Table
function renderTokensTable(tokens) {
    if (!tokens || tokens.length === 0) return;
    let html = `
        <table class="nebula-table">
            <thead>
                <tr>
                    <th>Index</th>
                    <th>Token Text</th>
                    <th>Token ID</th>
                    <th>Hex Bytes</th>
                    <th>Char Range</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
    `;
    tokens.forEach(t => {
        html += `
            <tr>
                <td>${t.index}</td>
                <td style="color: #06b6d4; font-weight: 700;">${t.text}</td>
                <td>${t.token_id}</td>
                <td style="color: #94a3b8;">${t.hex}</td>
                <td>[${t.start_char}..${t.end_char}]</td>
                <td><span style="font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; background: ${t.is_special ? 'rgba(244, 63, 94, 0.2)' : 'rgba(99, 102, 241, 0.2)'}; color: ${t.is_special ? '#f43f5e' : '#6366f1'};">${t.is_special ? 'Special' : 'Byte/Subword'}</span></td>
            </tr>
        `;
    });
    html += '</tbody></table>';
    el.tokensTable.innerHTML = html;
}

// Render KV Cache
function renderKVCache(kv) {
    if (!kv) return;
    el.kvMetrics.innerHTML = `
        <div class="metric-card">
            <span class="metric-label">CACHED TOKENS</span>
            <span class="metric-value">${kv.total_cached_tokens}</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">CACHE MEMORY</span>
            <span class="metric-value">${kv.memory_kb} KB</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">CURRENT TOKEN STEP</span>
            <span class="metric-value">${kv.step}</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">ACTIVE LAYERS</span>
            <span class="metric-value">${kv.layer_summaries ? kv.layer_summaries.length : 0}</span>
        </div>
    `;

    if (kv.layer_summaries) {
        let rows = '';
        kv.layer_summaries.forEach(l => {
            rows += `
                <div class="kv-layer-row">
                    <span class="kv-layer-label">Layer ${l.layer_idx}</span>
                    <div class="kv-bar-track">
                        <div class="kv-bar-fill" style="width: ${Math.min(100, l.cache_fill_percentage * 2)}%;"></div>
                    </div>
                    <span style="font-family: var(--font-mono); font-size: 0.8rem; color: #94a3b8;">${l.k_norm} K-norm</span>
                </div>
            `;
        });
        el.kvMatrixView.innerHTML = rows;
    }
}

// Render MoE Routing
function renderMoE(moeList) {
    if (!moeList || moeList.length === 0) {
        el.moeViewContainer.innerHTML = '<div class="empty-hint">No Mixture of Experts layers on this forward pass.</div>';
        return;
    }
    let html = '';
    moeList.forEach(m => {
        html += `
            <div class="moe-layer-card">
                <div class="moe-header">
                    <span class="moe-title">MoE Layer ${m.layer_idx}</span>
                    <span style="font-size: 0.75rem; color: #10b981;">Top-2 Experts Selected</span>
                </div>
                <div class="expert-chips-row">
        `;
        [0, 1, 2, 3].forEach(expIdx => {
            const isChosen = m.top_k_indices.includes(expIdx);
            const chosenIdx = m.top_k_indices.indexOf(expIdx);
            const weight = chosenIdx !== -1 ? m.top_k_weights[chosenIdx] : 0.0;
            html += `
                <div class="expert-chip ${isChosen ? 'chosen' : ''}">
                    <span class="expert-chip-name">Expert ${expIdx}</span>
                    <span class="expert-chip-weight">${isChosen ? (weight * 100).toFixed(1) + '%' : 'Inactive'}</span>
                </div>
            `;
        });
        html += `
                </div>
            </div>
        `;
    });
    el.moeViewContainer.innerHTML = html;
}

// Render Attention Heads
function renderAttention(attnList) {
    if (!attnList || attnList.length === 0) return;
    let html = '';
    attnList.forEach(layer => {
        layer.heads.forEach(h => {
            html += `
                <div class="head-card">
                    <div class="head-title">L${layer.layer_idx} Head ${h.head_idx}</div>
                    <div class="head-classification">${h.classification}</div>
                    <div class="head-entropy">Entropy: ${h.mean_entropy}</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Anchor: ${(h.bos_attention * 100).toFixed(0)}%</div>
                </div>
            `;
        });
    });
    el.attentionHeadsContainer.innerHTML = html;
}

// Render Logit Lens
function renderLogitLens(lensList) {
    if (!lensList || lensList.length === 0) return;
    let html = '';
    lensList.forEach(l => {
        const top = l.top_prediction || {};
        html += `
            <div class="lens-layer-card">
                <span class="lens-layer-name">${l.layer_name}</span>
                <div class="lens-top-pred">
                    <span class="pred-token">${top.token_str || 'N/A'}</span>
                    <span class="pred-prob">${(top.probability * 100).toFixed(1)}% prob</span>
                </div>
                <span style="font-family: var(--font-mono); font-size: 0.8rem; color: #64748b;">Entropy: ${l.entropy}</span>
            </div>
        `;
    });
    el.lensTimelineContainer.innerHTML = html;
}

// Render Neurons
function renderNeurons(neuronList) {
    if (!neuronList || neuronList.length === 0) return;
    let html = '';
    neuronList.forEach(n => {
        html += `
            <div class="moe-layer-card" style="margin-bottom: 12px;">
                <div class="moe-header">
                    <span class="moe-title">Layer ${n.layer_idx} Neurons (${n.active_neurons} Active / ${n.total_neurons})</span>
                    <span style="font-size: 0.75rem; color: #06b6d4;">Sparsity: ${n.sparsity_pct}%</span>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        `;
        n.top_neurons.slice(0, 8).forEach(nr => {
            html += `
                <span style="background: #151d2e; border: 1px solid #222e44; padding: 4px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: 0.75rem;">
                    N${nr.neuron_idx}: ${nr.max_activation}
                </span>
            `;
        });
        html += `
                </div>
            </div>
        `;
    });
    el.neuronExplorerContainer.innerHTML = html;
}

// Setup Activation Patching Studio
function setupPatchingStudio() {
    el.btnApplyPatch.addEventListener('click', async () => {
        el.btnApplyPatch.disabled = true;
        el.btnApplyPatch.textContent = 'Evaluating Intervention...';

        const payload = {
            prompt: el.promptInput.value,
            patches: [
                {
                    layer_idx: parseInt(document.getElementById('patch-layer').value),
                    target: document.getElementById('patch-target').value,
                    action: document.getElementById('patch-action').value,
                    scalar_multiplier: parseFloat(document.getElementById('patch-multiplier').value),
                }
            ]
        };

        try {
            const res = await fetch('/api/patch-compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            renderPatchDiffResults(data);
        } catch (e) {
            alert('Patch error: ' + e.message);
        } finally {
            el.btnApplyPatch.disabled = false;
            el.btnApplyPatch.textContent = 'Run Counterfactual Comparison';
        }
    });
}

function renderPatchDiffResults(data) {
    let html = `
        <div style="margin-bottom: 12px; font-weight: 700; color: ${data.primary_prediction_changed ? '#f43f5e' : '#10b981'};">
            ${data.primary_prediction_changed ? '⚠ PREDICTION SHIFT DETECTED: Counterfactual intervention altered top token choice!' : '✓ Decision Stable: Top token remained unchanged.'}
        </div>
        <div class="diff-grid">
            <div class="diff-col">
                <h4>BASELINE CANDIDATES</h4>
    `;
    data.baseline_top_candidates.forEach(c => {
        html += `<div style="font-family: var(--font-mono); font-size: 0.85rem; margin-bottom: 4px;">${c.token_str} (${(c.prob * 100).toFixed(1)}%)</div>`;
    });
    html += `
            </div>
            <div class="diff-col">
                <h4>COUNTERFACTUAL CANDIDATES</h4>
    `;
    data.patched_top_candidates.forEach(c => {
        html += `<div style="font-family: var(--font-mono); font-size: 0.85rem; margin-bottom: 4px; color: #06b6d4;">${c.token_str} (${(c.prob * 100).toFixed(1)}%)</div>`;
    });
    html += `
            </div>
        </div>
    `;
    el.patchDiffContainer.innerHTML = html;
}

// Setup Prompt Diff Studio
function setupPromptDiffStudio() {
    el.btnComparePrompts.addEventListener('click', async () => {
        el.btnComparePrompts.disabled = true;
        el.btnComparePrompts.textContent = 'Analyzing Prompts...';

        const payload = {
            prompt_a: el.diffPromptA.value,
            prompt_b: el.diffPromptB.value,
        };

        try {
            const res = await fetch('/api/prompt-impact', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            renderPromptDiffResults(data);
        } catch (e) {
            alert('Prompt compare error: ' + e.message);
        } finally {
            el.btnComparePrompts.disabled = false;
            el.btnComparePrompts.textContent = 'Compare Internal Pathways';
        }
    });
}

function renderPromptDiffResults(data) {
    let html = `
        <div class="diff-grid" style="margin-bottom: 16px;">
            <div class="diff-col">
                <h4>${data.prompt_a.label} (${data.prompt_a.tokens} tokens)</h4>
                <div style="font-family: var(--font-mono); font-size: 0.85rem;">Top Next: ${data.prompt_a.top_predictions[0]?.token_str} (${(data.prompt_a.top_predictions[0]?.probability * 100).toFixed(1)}%)</div>
            </div>
            <div class="diff-col">
                <h4>${data.prompt_b.label} (${data.prompt_b.tokens} tokens)</h4>
                <div style="font-family: var(--font-mono); font-size: 0.85rem; color: #06b6d4;">Top Next: ${data.prompt_b.top_predictions[0]?.token_str} (${(data.prompt_b.top_predictions[0]?.probability * 100).toFixed(1)}%)</div>
            </div>
        </div>
        <h4 style="font-size: 0.85rem; margin-bottom: 8px;">RESIDUAL STREAM SIMILARITY ACROSS DEPTH</h4>
    `;
    data.layer_divergence.forEach(l => {
        html += `
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px; font-family: var(--font-mono);">
                <span>${l.layer_name}</span>
                <span>Cosine Sim: ${l.cosine_similarity} | L2 Distance: ${l.l2_distance}</span>
            </div>
        `;
    });
    el.promptDiffContainer.innerHTML = html;
}

// Backward Studio
function setupBackwardStudio() {
    const btn = document.getElementById('btn-run-backward');
    const container = document.getElementById('backward-results-container');
    if (!btn || !container) return;

    btn.addEventListener('click', async () => {
        btn.disabled = true;
        btn.textContent = 'Computing Gradients...';
        try {
            const res = await fetch('/api/backward', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: el.promptInput.value }),
            });
            const data = await res.json();
            let html = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px; font-weight: 700;">
                    <span>Target Token: <span style="color: #06b6d4;">${data.target_token_str}</span> (ID: ${data.target_token_id})</span>
                    <span>Loss: <span style="color: #f59e0b;">${data.loss}</span></span>
                    <span>Status: <span style="color: #10b981;">${data.gradient_health}</span></span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px;">
            `;
            data.layer_gradients.forEach(lg => {
                html += `
                    <div class="grad-layer-card">
                        <span>Layer ${lg.layer_idx} ${lg.is_moe ? '(MoE)' : '(Dense)'}</span>
                        <div style="display: flex; gap: 16px; font-family: var(--font-mono); font-size: 0.8rem;">
                            <span>Residual Norm: ${lg.residual_grad_norm}</span>
                            <span>Attn Norm: ${lg.attn_grad_norm}</span>
                            <span>FFN Norm: ${lg.ffn_or_moe_grad_norm}</span>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            container.innerHTML = html;
        } catch (e) {
            alert('Backward error: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Compute Analytical Backward Pass';
        }
    });
}

// DLA Studio
function setupDLAStudio() {
    const btn = document.getElementById('btn-run-dla');
    const container = document.getElementById('dla-results-container');
    if (!btn || !container) return;

    btn.addEventListener('click', async () => {
        btn.disabled = true;
        btn.textContent = 'Decomposing Logits...';
        try {
            const res = await fetch('/api/dla', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: el.promptInput.value }),
            });
            const data = await res.json();
            let html = `
                <div style="margin-bottom: 12px; font-weight: 700;">
                    Target Token: <span style="color: #06b6d4;">${data.target_token_str}</span> | Baseline Logit: ${data.baseline_logit} | Embedding: ${data.embedding_contribution}
                </div>
                <h4 style="font-size: 0.85rem; margin-bottom: 8px; color: #10b981;">TOP PROMOTING HEADS</h4>
                <div style="margin-bottom: 16px;">
            `;
            data.top_promoting_heads.forEach(h => {
                html += `<span class="dla-head-pill promoter">L${h.layer_idx} H${h.head_idx}: +${h.direct_logit_contribution}</span>`;
            });
            html += `
                </div>
                <h4 style="font-size: 0.85rem; margin-bottom: 8px; color: #f43f5e;">TOP SUPPRESSING HEADS</h4>
                <div style="margin-bottom: 16px;">
            `;
            data.top_suppressing_heads.forEach(h => {
                html += `<span class="dla-head-pill suppressor">L${h.layer_idx} H${h.head_idx}: ${h.direct_logit_contribution}</span>`;
            });
            html += '</div>';
            container.innerHTML = html;
        } catch (e) {
            alert('DLA error: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Compute Direct Logit Attribution';
        }
    });
}

// Steering Studio
function setupSteeringStudio() {
    const btn = document.getElementById('btn-apply-steering');
    const slider = document.getElementById('steer-intensity');
    const sliderVal = document.getElementById('val-steer-intensity');
    const conceptSel = document.getElementById('concept-select');
    const container = document.getElementById('steering-results-container');
    if (!btn || !container) return;

    if (slider && sliderVal) {
        slider.addEventListener('input', () => { sliderVal.textContent = slider.value; });
    }

    btn.addEventListener('click', async () => {
        btn.disabled = true;
        btn.textContent = 'Steering Representation...';
        try {
            const res = await fetch('/api/steering/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: el.promptInput.value,
                    concept_name: conceptSel.value,
                    intensity: parseFloat(slider.value),
                }),
            });
            const data = await res.json();
            let html = `
                <div style="margin-bottom: 12px; font-weight: 700; color: ${data.prediction_flipped ? '#f59e0b' : '#10b981'};">
                    ${data.prediction_flipped ? '✦ PREDICTION SHIFTED by concept vector injection!' : 'Concept aligned: output remains stable.'}
                </div>
                <div class="diff-grid" style="margin-bottom: 16px;">
                    <div class="diff-col">
                        <h4>BASELINE PREDICTION</h4>
                        <div style="font-family: var(--font-mono); font-size: 0.85rem;">${data.baseline_prediction.token_str} (${(data.baseline_prediction.prob * 100).toFixed(1)}%)</div>
                        <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Alignment: ${data.concept_alignment_before}</div>
                    </div>
                    <div class="diff-col">
                        <h4>STEERED PREDICTION</h4>
                        <div style="font-family: var(--font-mono); font-size: 0.85rem; color: #06b6d4;">${data.steered_prediction.token_str} (${(data.steered_prediction.prob * 100).toFixed(1)}%)</div>
                        <div style="font-size: 0.75rem; color: #06b6d4; margin-top: 4px;">Alignment: ${data.concept_alignment_after}</div>
                    </div>
                </div>
            `;
            container.innerHTML = html;
        } catch (e) {
            alert('Steering error: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Apply Steering Vector';
        }
    });
}

// Start application
window.addEventListener('DOMContentLoaded', init);
