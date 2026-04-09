const BASE = 'http://127.0.0.1:5005';

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
    probFill:       document.getElementById('prob-fill'),
    probLiteral:    document.getElementById('prob-literal'),
    shapContainer:  document.getElementById('shap-container'),
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
    randomizeBtn:     document.getElementById('randomize-btn'),
    presetStruggling: document.getElementById('preset-struggling'),
    presetAchiever:   document.getElementById('preset-achiever'),
    presetBurnout:    document.getElementById('preset-burnout'),
};

let charts = { contrib: null, radar: null, sensitivity: null };

function initCharts() {
    Chart.defaults.font.family = "'Raleway', sans-serif";
    Chart.defaults.color = '#a1a1aa';
    const G = '#27272a';

    charts.contrib = new Chart(document.getElementById('contribChart'), {
        type: 'bar',
        data: {
            labels: ['Effort', 'Commitment', 'Wellness'],
            datasets: [{ data: [0,0,0], backgroundColor: '#52525b', barThickness: 16 }]
        },
        options: {
            indexAxis: 'y', responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { x: { grid: { color: G } }, y: { grid: { display: false } } }
        }
    });

    charts.radar = new Chart(document.getElementById('radarChart'), {
        type: 'radar',
        data: {
            labels: ['Effort', 'Commitment', 'Wellness'],
            datasets: [
                { label: 'Current', data: [0,0,0], backgroundColor: 'rgba(96,165,250,0.2)', borderColor: '#60a5fa', borderWidth: 2 },
                { label: 'Baseline', data: [50,50,50], backgroundColor: 'transparent', borderColor: '#3f3f46', borderDash: [5,5] }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { r: { grid: { color: G }, angleLines: { color: G }, ticks: { display: false }, min: 0, max: 100 } }
        }
    });

    charts.sensitivity = new Chart(document.getElementById('sensitivityChart'), {
        type: 'line',
        data: {
            labels: ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],
            datasets: [
                { label: 'Effort Influence', data: [], borderColor: '#4ade80', tension: 0.4, pointRadius: 0 },
                { label: 'Commitment Influence', data: [], borderColor: '#a78bfa', tension: 0.4, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { min: 0, max: 1, grid: { color: G } }, x: { grid: { color: G } } }
        }
    });
}

function renderShapPlot(c) {
    const total = Math.abs(c.studytime) + Math.abs(c.attendance) + Math.abs(c.wellbeing);
    const items = [{ name: 'Effort', v: c.studytime }, { name: 'Commit', v: c.attendance }, { name: 'Wellness', v: c.wellbeing }];
    dom.shapContainer.innerHTML = items.map(it => {
        const w = Math.max((Math.abs(it.v) / total) * 100, 1) || 33;
        return `<div style="width:${w}%" class="h-full ${it.v >= 0 ? 'bg-blue-600' : 'bg-red-600'} border-r border-zinc-900 flex items-center justify-center transition-all duration-300">
            <span class="text-[9px] font-bold text-white uppercase opacity-80">${it.name}</span>
        </div>`;
    }).join('');
}

async function refresh() {
    const payload = {
        studytime: parseFloat(dom.studytime.value),
        wellbeing: parseFloat(dom.wellbeing.value),
        attendance_rate: parseFloat(dom.absences.value)
    };
    
    dom.studytimeVal.textContent = payload.studytime;
    dom.wellbeingVal.textContent = payload.wellbeing;
    dom.absencesVal.textContent = payload.attendance_rate;

    try {
        const [p, e] = await Promise.all([
            fetch(`${BASE}/predict`, { method: 'POST', body: JSON.stringify(payload), headers: {'Content-Type': 'application/json'} }).then(r => r.json()),
            fetch(`${BASE}/explain`, { method: 'POST', body: JSON.stringify(payload), headers: {'Content-Type': 'application/json'} }).then(r => r.json())
        ]);
        
        dom.predictionText.textContent = p.prediction;
        dom.predictionText.className = `text-5xl font-semibold tracking-tight ${p.prediction === 'Pass' ? 'text-green-400' : 'text-red-400'}`;
        dom.confidenceText.textContent = (p.confidence * 100).toFixed(1) + '%';
        
        const pp = p.pass_prob * 100;
        dom.probFill.style.width = pp + '%';
        dom.probLiteral.textContent = pp.toFixed(1) + '%';
        dom.probFill.className = `${pp >= 50 ? 'bg-green-500' : 'bg-red-500'} h-full transition-all duration-300`;

        dom.reasoningText.innerHTML = e.reasoning;
        dom.actionPlanText.innerHTML = e.action_plan || "Candidate is currently within stable parameters.";
        
        dom.simpleText.innerHTML = `<div class='space-y-4'><div>${e.reasoning}</div><div class='bg-zinc-900/50 p-4 border border-zinc-800 rounded'>${e.action_plan || 'No immediate intervention required.'}</div></div>`;
        dom.detailedText.innerHTML = `<div class='font-mono text-[11px] leading-relaxed text-zinc-400'>${e.technical_reasoning}</div>`;

        const c = e.contributions;
        charts.contrib.data.datasets[0].data = [c.studytime, c.attendance, c.wellbeing];
        charts.contrib.data.datasets[0].backgroundColor = [
            c.studytime >= 0 ? '#3b82f6' : '#ef4444',
            c.attendance >= 0 ? '#3b82f6' : '#ef4444',
            c.wellbeing >= 0 ? '#3b82f6' : '#ef4444'
        ];
        charts.contrib.update();

        charts.radar.data.datasets[0].data = [ (payload.studytime/4)*100, payload.attendance_rate, (payload.wellbeing/5)*100 ];
        charts.radar.update();

        charts.sensitivity.data.datasets[0].data = e.sensitivity.studytime;
        charts.sensitivity.data.datasets[1].data = e.sensitivity.attendance;
        charts.sensitivity.update();
        renderShapPlot(c);

    } catch (err) { console.error('API Error:', err); }
}

const db = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };
const dr = db(refresh, 150);

[dom.studytime, dom.wellbeing, dom.absences].forEach(el => el.addEventListener('input', dr));
dom.presetStruggling.addEventListener('click', () => { dom.studytime.value=1; dom.wellbeing.value=1; dom.absences.value=0; refresh(); });
dom.presetAchiever.addEventListener('click',   () => { dom.studytime.value=4; dom.wellbeing.value=5; dom.absences.value=98; refresh(); });
dom.presetBurnout.addEventListener('click',    () => { dom.studytime.value=4; dom.wellbeing.value=1; dom.absences.value=85; refresh(); });
dom.randomizeBtn.addEventListener('click', () => { 
    dom.studytime.value=Math.ceil(Math.random()*4); dom.wellbeing.value=Math.ceil(Math.random()*5); dom.absences.value=Math.floor(Math.random()*100); refresh(); 
});

dom.deepAnalyticsBtn.addEventListener('click', () => { dom.analyticsModal.classList.remove('hidden','opacity-0'); dom.analyticsModal.classList.add('flex'); });
document.getElementById('close-analytics').addEventListener('click', () => { dom.analyticsModal.classList.add('hidden','opacity-0'); dom.analyticsModal.classList.remove('flex'); });
dom.explainBtn.addEventListener('click', () => { dom.explanationModal.classList.remove('hidden','opacity-0'); dom.explanationModal.classList.add('flex'); });
document.getElementById('close-modal').addEventListener('click', () => { dom.explanationModal.classList.add('hidden','opacity-0'); dom.explanationModal.classList.remove('flex'); });

dom.toggleSimple.addEventListener('click', () => {
    dom.toggleSimple.classList.add('bg-zinc-800','text-white'); dom.toggleTechnical.classList.remove('bg-zinc-800','text-white');
    dom.detailedText.classList.add('hidden'); dom.simpleText.classList.remove('hidden');
});
dom.toggleTechnical.addEventListener('click', () => {
    dom.toggleTechnical.classList.add('bg-zinc-800','text-white'); dom.toggleSimple.classList.remove('bg-zinc-800','text-white');
    dom.simpleText.classList.add('hidden'); dom.detailedText.classList.remove('hidden');
});

initCharts();
setTimeout(refresh, 500);
