// ── Config ────────────────────────────────────────────────────────────────────
// Use relative URLs so this works both locally (Flask dev) and on Railway
const BASE = '';

// ── State ─────────────────────────────────────────────────────────────────────
let selectedModel = 'Logistic Regression';
let lastPayload   = { studytime: 2, wellbeing: 3, attendance_rate: 100 };
let lastExplain   = null;
let abortCtrl     = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dom = {
    studytime:      document.getElementById('studytime'),
    wellbeing:      document.getElementById('wellbeing'),
    absences:       document.getElementById('absences'),
    studytimeVal:   document.getElementById('studytime-val'),
    wellbeingVal:   document.getElementById('wellbeing-val'),
    absencesVal:    document.getElementById('absences-val'),
    predictionText: document.getElementById('prediction-text'),
    confidenceText: document.getElementById('confidence-text'),
    reasoningText:  document.getElementById('reasoning-text'),
    actionPlanText: document.getElementById('action-plan-text'),
    counterfactualText: document.getElementById('counterfactual-text'),
    probFill:       document.getElementById('prob-fill'),
    probLiteral:    document.getElementById('prob-literal'),
    shapContainer:  document.getElementById('shap-container'),
    contribLabel:   document.getElementById('contrib-chart-label'),
    activeModelBadge: document.getElementById('active-model-badge'),
    modelSelect:    document.getElementById('model-select'),
    modelStatus:    document.getElementById('model-status'),

    explainBtn:       document.getElementById('explain-btn'),
    explanationModal: document.getElementById('explanation-modal'),
    closeModalBtn:    document.getElementById('close-modal'),
    simpleText:       document.getElementById('simple-explanation-text'),
    detailedText:     document.getElementById('detailed-explanation-text'),
    toggleSimple:     document.getElementById('toggle-simple'),
    toggleTechnical:  document.getElementById('toggle-technical'),

    analyticsModal:   document.getElementById('analytics-modal'),
    deepAnalyticsBtn: document.getElementById('deep-analytics-btn'),
    closeAnalyticsBtn:document.getElementById('close-analytics'),

    comparisonModal:  document.getElementById('comparison-modal'),
    modelCompareBtn:  document.getElementById('model-compare-btn'),
    closeComparison:  document.getElementById('close-comparison'),
    metricsTable:     document.getElementById('metrics-table'),
    loadXaiBtn:       document.getElementById('load-xai-btn'),
    xaiLoading:       document.getElementById('xai-loading'),
    xaiError:         document.getElementById('xai-error'),
    xaiModelLabel:    document.getElementById('xai-model-label'),
    compareAllBtn:    document.getElementById('compare-all-btn'),
    compareLoading:   document.getElementById('compare-loading'),
    compareResults:   document.getElementById('compare-results'),

    xaiModal:         document.getElementById('xai-modal'),
    xaiCompareBtn:    document.getElementById('xai-compare-btn'),
    closeXaiModal:    document.getElementById('close-xai-modal'),
    xaiModalSubtitle: document.getElementById('xai-modal-subtitle'),
    xaiModalLoading:  document.getElementById('xai-modal-loading'),
    xaiModalContent:  document.getElementById('xai-modal-content'),
    xaiValuesGrid:    document.getElementById('xai-values-grid'),

    randomizeBtn:     document.getElementById('randomize-btn'),
    presetStruggling: document.getElementById('preset-struggling'),
    presetAchiever:   document.getElementById('preset-achiever'),
    presetBurnout:    document.getElementById('preset-burnout'),
};

// ── Charts ────────────────────────────────────────────────────────────────────
let charts = {
    contrib: null, radar: null, sensitivity: null,
    xaiShap: null, xaiLime: null, xaiCompare: null, xaiModal: null,
};

const GRID_COLOR = '#27272a';

function initCharts() {
    Chart.defaults.font.family = "'Raleway', sans-serif";
    Chart.defaults.color = '#a1a1aa';

    charts.contrib = new Chart(document.getElementById('contribChart'), {
        type: 'bar',
        data: {
            labels: ['Effort', 'Commitment', 'Wellness'],
            datasets: [{ data: [0,0,0], backgroundColor: '#52525b', barThickness: 16 }]
        },
        options: {
            indexAxis: 'y', responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { x: { grid: { color: GRID_COLOR } }, y: { grid: { display: false } } }
        }
    });

    charts.radar = new Chart(document.getElementById('radarChart'), {
        type: 'radar',
        data: {
            labels: ['Effort', 'Commitment', 'Wellness'],
            datasets: [
                { label: 'Current',  data: [0,0,0],    backgroundColor: 'rgba(96,165,250,0.2)', borderColor: '#60a5fa', borderWidth: 2 },
                { label: 'Baseline', data: [50,50,50], backgroundColor: 'transparent', borderColor: '#3f3f46', borderDash: [5,5] }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { r: { grid: { color: GRID_COLOR }, angleLines: { color: GRID_COLOR }, ticks: { display: false }, min: 0, max: 100 } }
        }
    });

    charts.sensitivity = new Chart(document.getElementById('sensitivityChart'), {
        type: 'line',
        data: {
            labels: ['0','10','20','30','40','50','60','70','80','90','100'],
            datasets: [
                { label: 'Effort Influence',     data: [], borderColor: '#4ade80', tension: 0.4, pointRadius: 0 },
                { label: 'Commitment Influence', data: [], borderColor: '#a78bfa', tension: 0.4, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { min: 0, max: 1, grid: { color: GRID_COLOR } }, x: { grid: { color: GRID_COLOR } } }
        }
    });
}

// ── Network helpers ───────────────────────────────────────────────────────────
function apiPost(endpoint, data, signal) {
    return fetch(`${BASE}${endpoint}`, {
        method: 'POST',
        body: JSON.stringify(data),
        headers: { 'Content-Type': 'application/json' },
        ...(signal ? { signal } : {}),
    }).then(r => r.json());
}

function apiGet(endpoint) {
    return fetch(`${BASE}${endpoint}`).then(r => r.json());
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function updatePredictionPanel(p) {
    dom.predictionText.textContent = p.prediction;
    dom.predictionText.className = `text-5xl font-semibold tracking-tight ${p.prediction === 'Pass' ? 'text-green-400' : 'text-red-400'}`;
    dom.confidenceText.textContent = (p.confidence * 100).toFixed(1) + '%';
    const pp = p.pass_prob * 100;
    dom.probFill.style.width = pp + '%';
    dom.probLiteral.textContent = pp.toFixed(1) + '%';
    dom.probFill.className = `${pp >= 50 ? 'bg-green-500' : 'bg-red-500'} h-full transition-all duration-300`;
    document.getElementById('confidence-warning').classList.toggle('hidden', !(pp > 40 && pp < 60));
    document.getElementById('anomaly-warning').classList.toggle('hidden', !p.is_anomaly);
}

function updateContribChart(vals, label) {
    dom.contribLabel.textContent = label || 'Feature Contributions';
    charts.contrib.data.datasets[0].data = [vals.studytime, vals.attendance, vals.wellbeing];
    charts.contrib.data.datasets[0].backgroundColor = [
        vals.studytime  >= 0 ? '#3b82f6' : '#ef4444',
        vals.attendance >= 0 ? '#3b82f6' : '#ef4444',
        vals.wellbeing  >= 0 ? '#3b82f6' : '#ef4444',
    ];
    charts.contrib.update();
}

function updateSensitivity(sens) {
    charts.sensitivity.data.datasets[0].data = sens.studytime;
    charts.sensitivity.data.datasets[1].data = sens.attendance;
    charts.sensitivity.update();
}

function renderShapPlot(c) {
    const total = Math.abs(c.studytime) + Math.abs(c.attendance) + Math.abs(c.wellbeing) || 1;
    const items = [
        { name: 'Effort', v: c.studytime },
        { name: 'Commit', v: c.attendance },
        { name: 'Wellness', v: c.wellbeing },
    ];
    dom.shapContainer.innerHTML = items.map(it => {
        const w = Math.max((Math.abs(it.v) / total) * 100, 1);
        return `<div style="width:${w}%" class="h-full ${it.v >= 0 ? 'bg-blue-600' : 'bg-red-600'} border-r border-zinc-900 flex items-center justify-center transition-all duration-300">
            <span class="text-[9px] font-bold text-white uppercase opacity-80">${it.name}</span>
        </div>`;
    }).join('');
}

// ── Main refresh ──────────────────────────────────────────────────────────────
async function refresh() {
    const payload = {
        studytime:       parseFloat(dom.studytime.value),
        wellbeing:       parseFloat(dom.wellbeing.value),
        attendance_rate: parseFloat(dom.absences.value),
    };
    lastPayload = payload;

    dom.studytimeVal.textContent  = payload.studytime;
    dom.wellbeingVal.textContent  = payload.wellbeing;
    dom.absencesVal.textContent   = payload.attendance_rate;
    dom.activeModelBadge.textContent = selectedModel;

    if (abortCtrl) abortCtrl.abort();
    abortCtrl = new AbortController();
    const sig = abortCtrl.signal;

    try {
        if (selectedModel === 'Logistic Regression') {
            const [p, e] = await Promise.all([
                apiPost('/predict', payload, sig),
                apiPost('/explain', payload, sig),
            ]);
            lastExplain = e;

            updatePredictionPanel(p);
            updateContribChart(e.contributions, 'LR Logit Contributions');
            renderShapPlot(e.contributions);
            updateSensitivity(e.sensitivity);

            dom.reasoningText.innerHTML  = e.reasoning;
            dom.actionPlanText.innerHTML = e.action_plan || 'Candidate is within stable parameters.';
            dom.counterfactualText.innerHTML = '';

            dom.simpleText.innerHTML = `<div class='space-y-4'><div>${e.reasoning}</div>
                <div class='bg-zinc-900/50 p-4 border border-zinc-800 rounded'>${e.action_plan || 'No immediate intervention required.'}</div></div>`;
            dom.detailedText.innerHTML = `<div class='font-mono text-[11px] leading-relaxed text-zinc-400'>${e.technical_reasoning}</div>`;

            charts.radar.data.datasets[0].data = [
                (payload.studytime / 4) * 100,
                payload.attendance_rate,
                (payload.wellbeing / 5) * 100
            ];
            charts.radar.update();

        } else {
            const fullPayload = { ...payload, model_name: selectedModel };

            const p = await apiPost('/predict', fullPayload, sig);
            updatePredictionPanel(p);

            dom.modelStatus.textContent = (selectedModel === 'SVM' || selectedModel === 'KNN')
                ? 'Computing SHAP (slow for SVM/KNN)...' : 'Computing...';

            const e = await apiPost('/explain/full', fullPayload, sig);
            dom.modelStatus.textContent = 'Select a model — SVM/KNN may be slower';

            const fnames = ['Academic Effort', 'Institutional Commitment', 'Wellness & Balance'];

            if (e.shap_values) {
                const sv = e.shap_values;
                const contrib = {
                    studytime:  sv[fnames[0]] ?? 0,
                    attendance: sv[fnames[1]] ?? 0,
                    wellbeing:  sv[fnames[2]] ?? 0,
                };
                updateContribChart(contrib, `SHAP Values — ${selectedModel}`);
                renderShapPlot(contrib);
            } else {
                updateContribChart({ studytime: 0, attendance: 0, wellbeing: 0 },
                    `${selectedModel} — SHAP unavailable`);
            }

            if (e.sensitivity) updateSensitivity(e.sensitivity);

            charts.radar.data.datasets[0].data = [
                (payload.studytime / 4) * 100,
                payload.attendance_rate,
                (payload.wellbeing / 5) * 100
            ];
            charts.radar.update();

            const pr   = p.prediction;
            const prob = (p.pass_prob * 100).toFixed(1);
            dom.reasoningText.innerHTML =
                `<span class="${pr === 'Pass' ? 'text-green-400 font-bold' : 'text-red-400 font-bold'}">${selectedModel} predicts ${pr}</span> with <b>${prob}%</b> pass probability.` +
                (e.shap_values ? ` SHAP attribution computed — see "SHAP vs LIME" for detail.` : '');

            const cf = e.counterfactual;
            if (cf && cf.flip_value !== null) {
                dom.actionPlanText.innerHTML = '';
                dom.counterfactualText.innerHTML =
                    `<b>Counterfactual:</b> Increasing <b>${cf.feature}</b> from ${cf.current_value} → <b>${cf.flip_value}</b> flips prediction to <span class='text-green-400'>Pass</span>.`;
            } else {
                dom.actionPlanText.innerHTML = cf ? 'No single-feature intervention found within range.' : '';
                dom.counterfactualText.innerHTML = '';
            }

            dom.simpleText.innerHTML = `<div class='space-y-3 text-sm'>
                <div>${dom.reasoningText.innerHTML}</div>
                <div class='bg-zinc-900/50 p-3 border border-zinc-800 rounded text-xs font-mono'>
                    <b>Model:</b> ${selectedModel}<br>
                    <b>P(Pass):</b> ${prob}%<br>
                    ${e.shap_values ? `<b>SHAP (Effort):</b> ${(e.shap_values[fnames[0]] || 0).toFixed(4)}<br>` : ''}
                    ${e.lime_weights ? `<b>LIME (Effort):</b> ${(e.lime_weights[fnames[0]] || 0).toFixed(4)}` : ''}
                </div></div>`;
            dom.detailedText.innerHTML = `<div class='font-mono text-[11px] text-zinc-400 leading-relaxed'>
                Model: ${selectedModel}<br>Prediction: ${pr} (${prob}%)<br>
                ${cf ? `Counterfactual flip at ${cf.feature} = ${cf.flip_value}` : ''}</div>`;
        }
    } catch (err) {
        if (err.name !== 'AbortError') console.error('API Error:', err);
        dom.modelStatus.textContent = 'Request failed — is the server running?';
    }
}

// ── Debounce ──────────────────────────────────────────────────────────────────
const db = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };
const dr = db(refresh, 150);

// ── Model selector ────────────────────────────────────────────────────────────
dom.modelSelect.addEventListener('change', () => {
    selectedModel = dom.modelSelect.value;
    dom.activeModelBadge.textContent = selectedModel;
    refresh();
});

// ── Sliders + personas ────────────────────────────────────────────────────────
[dom.studytime, dom.wellbeing, dom.absences].forEach(el => el.addEventListener('input', dr));

dom.presetStruggling.addEventListener('click', () => { dom.studytime.value=1; dom.wellbeing.value=1; dom.absences.value=0; refresh(); });
dom.presetAchiever.addEventListener('click',   () => { dom.studytime.value=4; dom.wellbeing.value=5; dom.absences.value=98; refresh(); });
dom.presetBurnout.addEventListener('click',    () => { dom.studytime.value=4; dom.wellbeing.value=1; dom.absences.value=85; refresh(); });
dom.randomizeBtn.addEventListener('click', () => {
    dom.studytime.value = Math.ceil(Math.random()*4);
    dom.wellbeing.value = Math.ceil(Math.random()*5);
    dom.absences.value  = Math.floor(Math.random()*100);
    refresh();
});

// ── Existing modals ───────────────────────────────────────────────────────────
dom.deepAnalyticsBtn.addEventListener('click', () => { dom.analyticsModal.classList.remove('hidden','opacity-0'); dom.analyticsModal.classList.add('flex'); });
document.getElementById('close-analytics').addEventListener('click', () => { dom.analyticsModal.classList.add('hidden','opacity-0'); dom.analyticsModal.classList.remove('flex'); });
dom.explainBtn.addEventListener('click', () => { dom.explanationModal.classList.remove('hidden','opacity-0'); dom.explanationModal.classList.add('flex'); });
document.getElementById('close-modal').addEventListener('click', () => { dom.explanationModal.classList.add('hidden','opacity-0'); dom.explanationModal.classList.remove('flex'); });

dom.toggleSimple.addEventListener('click', () => {
    dom.toggleSimple.classList.add('bg-zinc-800','text-white');
    dom.toggleTechnical.classList.remove('bg-zinc-800','text-white');
    dom.detailedText.classList.add('hidden');
    dom.simpleText.classList.remove('hidden');
});
dom.toggleTechnical.addEventListener('click', () => {
    dom.toggleTechnical.classList.add('bg-zinc-800','text-white');
    dom.toggleSimple.classList.remove('bg-zinc-800','text-white');
    dom.simpleText.classList.add('hidden');
    dom.detailedText.classList.remove('hidden');
});

// ── Comparison modal ──────────────────────────────────────────────────────────
dom.modelCompareBtn.addEventListener('click', () => {
    dom.comparisonModal.classList.remove('hidden','opacity-0');
    dom.comparisonModal.classList.add('flex');
    dom.xaiModelLabel.textContent = selectedModel;
    loadMetricsTable();
});
dom.closeComparison.addEventListener('click', () => {
    dom.comparisonModal.classList.add('hidden','opacity-0');
    dom.comparisonModal.classList.remove('flex');
});

async function loadMetricsTable() {
    dom.metricsTable.innerHTML = '<div class="text-zinc-500 text-xs p-3">Loading...</div>';
    try {
        const data = await apiGet('/models');
        if (data.error) {
            dom.metricsTable.innerHTML = `<div class="text-amber-400 text-xs p-3">${data.error}</div>`;
            return;
        }
        const rows = data.models.map(m => {
            const isActive = m.name === selectedModel;
            return `<tr class="border-b border-zinc-800 hover:bg-zinc-900/40 ${isActive ? 'bg-purple-900/10' : ''}">
                <td class="tbl-cell font-medium ${isActive ? 'text-purple-400' : 'text-zinc-300'}">${m.name}${isActive ? ' ✦' : ''}</td>
                <td class="tbl-cell font-mono text-zinc-300">${(m.accuracy*100).toFixed(2)}%</td>
                <td class="tbl-cell font-mono text-zinc-300">${(m.f1*100).toFixed(2)}%</td>
                <td class="tbl-cell font-mono text-zinc-300">${m.auc_roc.toFixed(4)}</td>
                <td class="tbl-cell font-mono text-zinc-300">${(m.cv_mean*100).toFixed(2)}% ± ${(m.cv_std*100).toFixed(2)}%</td>
                <td class="tbl-cell">
                    <button onclick="selectModelFromTable('${m.name}')" class="text-[10px] bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-2 py-0.5 rounded cursor-pointer tracking-wide uppercase">Select</button>
                </td>
            </tr>`;
        }).join('');
        dom.metricsTable.innerHTML = `
            <table class="w-full text-left border border-zinc-800 rounded overflow-hidden">
                <thead class="compare-hdr">
                    <tr class="text-[10px] uppercase tracking-widest text-zinc-500 border-b border-zinc-700">
                        <th class="tbl-cell">Model</th>
                        <th class="tbl-cell">Accuracy</th>
                        <th class="tbl-cell">F1 (macro)</th>
                        <th class="tbl-cell">AUC-ROC</th>
                        <th class="tbl-cell">5-Fold CV</th>
                        <th class="tbl-cell"></th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>`;
    } catch (err) {
        dom.metricsTable.innerHTML = '<div class="text-red-400 text-xs p-3">Failed to load. Server running? train_all.py run?</div>';
    }
}

function selectModelFromTable(name) {
    selectedModel = name;
    dom.modelSelect.value = name;
    dom.activeModelBadge.textContent = name;
    dom.xaiModelLabel.textContent = name;
    refresh();
    loadMetricsTable();
}

// ── XAI in comparison modal ───────────────────────────────────────────────────
dom.loadXaiBtn.addEventListener('click', loadXaiForCurrentModel);

async function loadXaiForCurrentModel() {
    dom.xaiLoading.classList.remove('hidden');
    dom.xaiError.textContent = '';
    dom.xaiModelLabel.textContent = selectedModel;

    const payload = { ...lastPayload, model_name: selectedModel };
    try {
        const [shapData, limeData] = await Promise.all([
            apiPost('/explain/shap', payload),
            apiPost('/explain/lime', payload),
        ]);

        if (shapData.error || limeData.error) {
            dom.xaiError.textContent = shapData.error || limeData.error;
            return;
        }

        const fnames = ['Academic Effort', 'Institutional Commitment', 'Wellness & Balance'];
        const labels = ['Effort', 'Commitment', 'Wellness'];
        const sv = fnames.map(f => shapData.shap_values ? (shapData.shap_values[f] ?? 0) : 0);
        const lv = fnames.map(f => limeData.lime_weights ? (limeData.lime_weights[f] ?? 0) : 0);

        renderXaiShapChart(labels, sv);
        renderXaiLimeChart(labels, lv);
        renderXaiGroupedChart(labels, sv, lv, null);

    } catch (err) {
        dom.xaiError.textContent = 'XAI computation failed: ' + err.message;
    } finally {
        dom.xaiLoading.classList.add('hidden');
    }
}

function _makeBarChart(canvasId, labels, datasets, old) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    if (old) old.destroy();
    return new Chart(canvas, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#a1a1aa', font: { size: 10 } } } },
            scales: {
                x: { grid: { color: GRID_COLOR }, ticks: { color: '#a1a1aa' } },
                y: { grid: { color: GRID_COLOR }, ticks: { color: '#a1a1aa' } },
            }
        }
    });
}

function renderXaiShapChart(labels, vals) {
    charts.xaiShap = _makeBarChart('xai-shap-chart', labels, [{
        label: 'SHAP', data: vals,
        backgroundColor: vals.map(v => v >= 0 ? 'rgba(59,130,246,0.8)' : 'rgba(239,68,68,0.8)'),
        borderColor: vals.map(v => v >= 0 ? '#3b82f6' : '#ef4444'), borderWidth: 1,
    }], charts.xaiShap);
}

function renderXaiLimeChart(labels, vals) {
    charts.xaiLime = _makeBarChart('xai-lime-chart', labels, [{
        label: 'LIME', data: vals,
        backgroundColor: vals.map(v => v >= 0 ? 'rgba(234,179,8,0.8)' : 'rgba(239,68,68,0.8)'),
        borderColor: vals.map(v => v >= 0 ? '#eab308' : '#ef4444'), borderWidth: 1,
    }], charts.xaiLime);
}

function renderXaiGroupedChart(labels, sv, lv, logitVals) {
    const datasets = [
        { label: 'SHAP', data: sv, backgroundColor: 'rgba(59,130,246,0.8)',  borderColor: '#3b82f6', borderWidth: 1 },
        { label: 'LIME', data: lv, backgroundColor: 'rgba(234,179,8,0.8)',   borderColor: '#eab308', borderWidth: 1 },
    ];
    if (logitVals) {
        datasets.push({ label: 'Logit (LR)', data: logitVals, backgroundColor: 'rgba(34,197,94,0.8)', borderColor: '#22c55e', borderWidth: 1 });
    }
    charts.xaiCompare = _makeBarChart('xai-comparison-chart', labels, datasets, charts.xaiCompare);
}

// ── Cross-model compare ───────────────────────────────────────────────────────
dom.compareAllBtn.addEventListener('click', runCompareAll);

async function runCompareAll() {
    dom.compareAllBtn.disabled = true;
    dom.compareLoading.classList.remove('hidden');
    dom.compareResults.innerHTML = '';

    try {
        const data = await apiPost('/explain/compare', lastPayload);
        if (data.error) {
            dom.compareResults.innerHTML = `<div class="text-red-400 text-xs p-3">${data.error}</div>`;
            return;
        }

        const fnames = ['Academic Effort', 'Institutional Commitment', 'Wellness & Balance'];
        const short  = ['Effort', 'Commit.', 'Wellness'];

        const colorVal = v => {
            if (Math.abs(v) < 0.01) return 'text-zinc-500';
            return v > 0 ? 'text-blue-400' : 'text-red-400';
        };

        const shapHdrs = short.map(n => `<th class="tbl-cell text-blue-400/80">SHAP: ${n}</th>`).join('');
        const limeHdrs = short.map(n => `<th class="tbl-cell text-yellow-400/80">LIME: ${n}</th>`).join('');

        const rows = data.results.map(r => {
            const sc = fnames.map(f => { const v = r.shap_values[f] ?? 0; return `<td class="tbl-cell font-mono ${colorVal(v)}">${v.toFixed(3)}</td>`; }).join('');
            const lc = fnames.map(f => { const v = r.lime_weights[f] ?? 0; return `<td class="tbl-cell font-mono ${colorVal(v)}">${v.toFixed(3)}</td>`; }).join('');
            const isActive = r.model_name === selectedModel;
            return `<tr class="border-b border-zinc-800 hover:bg-zinc-900/30 ${isActive ? 'bg-purple-900/10' : ''}">
                <td class="tbl-cell font-medium ${isActive ? 'text-purple-400' : 'text-zinc-300'}">${r.model_name}</td>
                <td class="tbl-cell font-semibold ${r.prediction === 'Pass' ? 'text-green-400' : 'text-red-400'}">${r.prediction}</td>
                <td class="tbl-cell font-mono text-zinc-400">${(r.pass_prob*100).toFixed(1)}%</td>
                ${sc}${lc}
            </tr>`;
        }).join('');

        dom.compareResults.innerHTML = `
            <table class="w-full text-left border border-zinc-800 rounded overflow-hidden">
                <thead class="compare-hdr">
                    <tr class="text-[10px] uppercase tracking-wider text-zinc-500 border-b border-zinc-700">
                        <th class="tbl-cell">Model</th><th class="tbl-cell">Pred.</th><th class="tbl-cell">P(Pass)</th>
                        ${shapHdrs}${limeHdrs}
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>`;
    } catch (err) {
        dom.compareResults.innerHTML = `<div class="text-red-400 text-xs p-3">Compare failed: ${err.message}</div>`;
    } finally {
        dom.compareAllBtn.disabled = false;
        dom.compareLoading.classList.add('hidden');
    }
}

// ── SHAP vs LIME quick modal ──────────────────────────────────────────────────
dom.xaiCompareBtn.addEventListener('click', openXaiModal);
dom.closeXaiModal.addEventListener('click', () => {
    dom.xaiModal.classList.add('hidden','opacity-0');
    dom.xaiModal.classList.remove('flex');
});

async function openXaiModal() {
    dom.xaiModal.classList.remove('hidden','opacity-0');
    dom.xaiModal.classList.add('flex');
    dom.xaiModalSubtitle.textContent = `Current model: ${selectedModel}`;
    dom.xaiModalLoading.classList.remove('hidden');
    dom.xaiModalContent.classList.add('hidden');

    const payload = { ...lastPayload, model_name: selectedModel };
    const fnames  = ['Academic Effort', 'Institutional Commitment', 'Wellness & Balance'];
    const labels  = ['Effort', 'Commitment', 'Wellness'];

    try {
        const [shapData, limeData] = await Promise.all([
            apiPost('/explain/shap', payload),
            apiPost('/explain/lime', payload),
        ]);

        const sv = shapData.shap_values || {};
        const lv = limeData.lime_weights || {};
        const logitSrc = (lastExplain && selectedModel === 'Logistic Regression') ? {
            [fnames[0]]: lastExplain.contributions.studytime,
            [fnames[1]]: lastExplain.contributions.attendance,
            [fnames[2]]: lastExplain.contributions.wellbeing,
        } : null;

        const shapVals  = fnames.map(f => sv[f] ?? 0);
        const limeVals  = fnames.map(f => lv[f] ?? 0);
        const logitVals = logitSrc ? fnames.map(f => logitSrc[f] ?? 0) : null;

        const datasets = [
            { label: 'SHAP', data: shapVals, backgroundColor: 'rgba(59,130,246,0.85)', borderColor: '#3b82f6', borderWidth: 1 },
            { label: 'LIME', data: limeVals, backgroundColor: 'rgba(234,179,8,0.85)',  borderColor: '#eab308', borderWidth: 1 },
        ];
        if (logitVals) {
            datasets.push({ label: 'Logit (LR)', data: logitVals, backgroundColor: 'rgba(34,197,94,0.85)', borderColor: '#22c55e', borderWidth: 1 });
        }
        charts.xaiModal = _makeBarChart('xai-modal-chart', labels, datasets, charts.xaiModal);

        dom.xaiValuesGrid.innerHTML = fnames.map((f, i) => `
            <div class="bg-zinc-900/50 border border-zinc-800 rounded p-3 flex flex-col gap-1.5">
                <div class="text-[10px] uppercase tracking-widest text-zinc-500 font-semibold">${labels[i]}</div>
                <div class="text-xs font-mono text-blue-400">SHAP: ${shapVals[i].toFixed(4)}</div>
                <div class="text-xs font-mono text-yellow-400">LIME: ${limeVals[i].toFixed(4)}</div>
                ${logitVals ? `<div class="text-xs font-mono text-green-400">Logit: ${logitVals[i].toFixed(4)}</div>` : ''}
            </div>`
        ).join('');

        dom.xaiModalLoading.classList.add('hidden');
        dom.xaiModalContent.classList.remove('hidden');

    } catch (err) {
        dom.xaiModalLoading.textContent = 'Failed: ' + err.message;
    }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
initCharts();
setTimeout(refresh, 500);
