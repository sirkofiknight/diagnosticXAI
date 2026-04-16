import os
import sys
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Make sibling modules (xai_shap, xai_lime) importable
_BACKEND = os.path.dirname(os.path.abspath(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

app = Flask(__name__,
            template_folder=os.path.join('..', 'templates'),
            static_folder=os.path.join('..', 'static'))
CORS(app)

LEGACY_PATH  = os.path.join(_BACKEND, 'expert_v3.pkl')
COMPARE_PATH = os.path.join(_BACKEND, 'models_comparison.pkl')


# ── Model loading ─────────────────────────────────────────────────────────────

def _train_lr_legacy():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    try:
        from ucimlrepo import fetch_ucirepo
        sp = fetch_ucirepo(id=320)
        X  = sp.data.features[['studytime', 'absences', 'health']].values.astype(float)
        y  = (sp.data.targets['G3'] >= 10).astype(int).values
    except Exception:
        np.random.seed(42)
        n = 2000
        X = np.column_stack([np.random.randint(1, 5, n).astype(float),
                              np.random.randint(0, 94, n).astype(float),
                              np.random.randint(1, 6, n).astype(float)])
        logit = 2.0*(X[:,0]-2.5) - 0.15*(X[:,1]-30) + 1.0*(X[:,2]-3.0)
        y = (logit > 0).astype(int)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    mdl    = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    mdl.fit(X_sc, y)
    fnames = ['Academic Effort', 'Institutional Commitment', 'Wellness & Balance']
    with open(LEGACY_PATH, 'wb') as f:
        pickle.dump({'model': mdl, 'scaler': scaler, 'feature_names': fnames}, f)
    return mdl, scaler, fnames


def _load_legacy():
    if os.path.exists(LEGACY_PATH):
        with open(LEGACY_PATH, 'rb') as f:
            d = pickle.load(f)
        return d['model'], d['scaler'], d['feature_names']
    return _train_lr_legacy()


def _load_comparison():
    if os.path.exists(COMPARE_PATH):
        with open(COMPARE_PATH, 'rb') as f:
            return pickle.load(f)
    return None


model, scaler, feature_names = _load_legacy()
u, s = scaler.mean_, scaler.scale_
comparison_data = _load_comparison()


# ── Request helpers ───────────────────────────────────────────────────────────

def _parse_input(req):
    st        = float(req.get('studytime', 1))
    wb        = float(req.get('wellbeing', 1))
    ar        = float(req.get('attendance_rate', 100))
    abs_count = (100 - ar) / 100 * 93
    X_raw     = np.array([[st, abs_count, wb]])
    return st, wb, ar, abs_count, X_raw


def _get_model_scaler(model_name):
    if model_name and model_name != 'Logistic Regression' and comparison_data:
        m  = comparison_data['models'].get(model_name)
        sc = comparison_data['scaler']
        if m is not None:
            return m, sc
    return model, scaler


# ── Core routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    req        = request.json
    model_name = req.get('model_name', 'Logistic Regression')
    st, wb, ar, abs_count, X_raw = _parse_input(req)
    m, sc = _get_model_scaler(model_name)
    X_sc  = sc.transform(X_raw)
    prob  = m.predict_proba(X_sc)[0]
    pred  = int(m.predict(X_sc)[0])
    return jsonify({
        'prediction': 'Pass' if pred == 1 else 'Fail',
        'confidence': float(prob[1] if pred == 1 else prob[0]),
        'pass_prob':  float(prob[1]),
        'is_anomaly': bool(np.max(np.abs(X_sc)) > 5.0),
        'model_name': model_name,
    })


@app.route('/explain', methods=['POST'])
def explain():
    """Backward-compatible LR logit decomposition endpoint."""
    req  = request.json
    st, wb, ar, abs_count, X_raw = _parse_input(req)
    X_sc = scaler.transform(X_raw)

    coef, intercept = model.coef_[0], model.intercept_[0]
    z    = X_sc[0]
    c    = coef * z
    lgit = float(intercept + np.sum(c))
    prob = 1 / (1 + np.exp(-lgit))

    dom_idx  = int(np.argmax(np.abs(c)))
    strongest = feature_names[dom_idx]

    if lgit > 0:
        human = (f"<span class='text-green-400 font-bold'>The model predicts a PASS</span> "
                 f"because your total diagnostic score ({lgit:+.2f}) is above the critical "
                 f"threshold. The most influential factor is your <b>{strongest}</b>, "
                 f"adding {c[dom_idx]:+.2f} to the success logit.")
    else:
        human = (f"<span class='text-red-400 font-bold'>The model predicts a FAIL</span> "
                 f"because your total diagnostic score ({lgit:+.2f}) is in the risk zone. "
                 f"The primary driver is your <b>{strongest}</b>. "
                 f"Probability of success is only {prob*100:.1f}%.")

    tech = (f"<b>Decision Logit Equation (L):</b><br>"
            f"L = β₀ + Σ(βᵢ · zᵢ)<br><br>"
            f"L = {intercept:.3f} (Intercept)<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{coef[0]:+.3f} × {z[0]:.2f} ({feature_names[0]})<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{coef[1]:+.3f} × {z[1]:.2f} ({feature_names[1]})<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{coef[2]:+.3f} × {z[2]:.2f} ({feature_names[2]})<br>"
            f"───────────────────<br>"
            f"<b>L = {lgit:.3f}</b><br><br>"
            f"<b>P(Pass) = σ(L) = 1/(1+e⁻ᴸ) = {prob:.4f}</b>")

    action = ""
    if lgit < 0:
        needed_z0 = (0.1 - intercept - coef[1]*z[1] - coef[2]*z[2]) / coef[0]
        needed_st = needed_z0 * s[0] + u[0]
        if 1 <= needed_st <= 6:
            action = (f"<b>Intervention Strategy:</b> Increasing <b>{feature_names[0]}</b> "
                      f"to level <b>{needed_st:.1f}</b> would flip the prediction to "
                      f"<span class='text-green-400'>PASS</span>.")
        else:
            action = ("<b>Intervention Strategy:</b> Indicators are severely depressed. "
                      "A holistic improvement across all three pillars is required.")

    return jsonify({
        'contributions': {
            'studytime':  float(c[0]),
            'attendance': float(c[1]),
            'wellbeing':  float(c[2]),
        },
        'reasoning':           human,
        'technical_reasoning': tech,
        'action_plan':         action,
        'sensitivity': {
            'studytime':  [float(model.predict_proba(scaler.transform([[v, abs_count, wb]]))[0][1])
                           for v in np.linspace(1, 4, 11)],
            'attendance': [float(model.predict_proba(scaler.transform([[st, (100-v)/100*93, wb]]))[0][1])
                           for v in np.linspace(0, 100, 11)],
        },
    })


# ── Multi-model & XAI routes ──────────────────────────────────────────────────

@app.route('/models', methods=['GET'])
def get_models():
    if not comparison_data:
        return jsonify({'error': 'models_comparison.pkl not found. '
                        'Run: python backend/train_all.py'}), 503
    rows = []
    for name, m in comparison_data['metrics'].items():
        rows.append({
            'name':      name,
            'accuracy':  round(m['accuracy'],  4),
            'precision': round(m['precision'], 4),
            'recall':    round(m['recall'],    4),
            'f1':        round(m['f1'],        4),
            'auc_roc':   round(m['auc_roc'],   4),
            'cv_mean':   round(m['cv_mean'],   4),
            'cv_std':    round(m['cv_std'],    4),
        })
    return jsonify({'models': rows})


@app.route('/explain/shap', methods=['POST'])
def explain_shap():
    if not comparison_data:
        return jsonify({'error': 'models_comparison.pkl not found.'}), 503
    from xai_shap import get_shap_values
    req        = request.json
    model_name = req.get('model_name', 'Logistic Regression')
    st, wb, ar, abs_count, X_raw = _parse_input(req)
    sc     = comparison_data['scaler']
    X_sc   = sc.transform(X_raw)
    X_bg   = comparison_data['X_train']
    fnames = comparison_data['feature_names']
    m = comparison_data['models'].get(model_name)
    if m is None:
        return jsonify({'error': f'Model "{model_name}" not found.'}), 404
    sv, bv = get_shap_values(m, X_sc, X_bg, model_name)
    return jsonify({
        'shap_values': {fnames[i]: float(sv[i]) for i in range(len(fnames))},
        'base_value':  float(bv),
        'model_name':  model_name,
    })


@app.route('/explain/lime', methods=['POST'])
def explain_lime():
    if not comparison_data:
        return jsonify({'error': 'models_comparison.pkl not found.'}), 503
    from xai_lime import get_lime_explanation
    req        = request.json
    model_name = req.get('model_name', 'Logistic Regression')
    st, wb, ar, abs_count, X_raw = _parse_input(req)
    sc          = comparison_data['scaler']
    X_train_raw = comparison_data['X_train_raw']
    fnames      = comparison_data['feature_names']
    m = comparison_data['models'].get(model_name)
    if m is None:
        return jsonify({'error': f'Model "{model_name}" not found.'}), 404
    result = get_lime_explanation(m, sc, X_raw[0], X_train_raw, fnames, model_name)
    return jsonify({
        'lime_weights': {fn: float(w) for fn, w in result},
        'model_name':   model_name,
    })


@app.route('/explain/compare', methods=['POST'])
def explain_compare():
    """SHAP + LIME + prediction for ALL 6 models on the same input."""
    if not comparison_data:
        return jsonify({'error': 'models_comparison.pkl not found.'}), 503
    from xai_shap import get_shap_values
    from xai_lime import get_lime_explanation

    req = request.json
    st, wb, ar, abs_count, X_raw = _parse_input(req)
    sc          = comparison_data['scaler']
    X_sc        = sc.transform(X_raw)
    X_train     = comparison_data['X_train']
    X_train_raw = comparison_data['X_train_raw']
    fnames      = comparison_data['feature_names']

    results = []
    for mname, m in comparison_data['models'].items():
        prob = m.predict_proba(X_sc)[0]
        pred = int(m.predict(X_sc)[0])
        sv, _  = get_shap_values(m, X_sc, X_train, mname)
        lr     = get_lime_explanation(m, sc, X_raw[0], X_train_raw, fnames, mname)
        ld     = dict(lr)
        results.append({
            'model_name':   mname,
            'prediction':   'Pass' if pred == 1 else 'Fail',
            'pass_prob':    float(prob[1]),
            'shap_values':  {fnames[i]: float(sv[i]) for i in range(len(fnames))},
            'lime_weights': {fn: float(ld.get(fn, 0.0)) for fn in fnames},
        })
    return jsonify({'results': results})


@app.route('/explain/full', methods=['POST'])
def explain_full():
    """Full XAI: SHAP + LIME + counterfactual + sensitivity for any model."""
    req        = request.json
    model_name = req.get('model_name', 'Logistic Regression')
    st, wb, ar, abs_count, X_raw = _parse_input(req)

    if comparison_data and model_name in comparison_data['models']:
        m, sc       = comparison_data['models'][model_name], comparison_data['scaler']
        X_train     = comparison_data['X_train']
        X_train_raw = comparison_data['X_train_raw']
        fnames      = comparison_data['feature_names']
    else:
        m, sc       = model, scaler
        X_train = X_train_raw = None
        fnames  = feature_names

    X_sc = sc.transform(X_raw)
    prob = m.predict_proba(X_sc)[0]
    pred = int(m.predict(X_sc)[0])

    resp = {
        'model_name': model_name,
        'prediction': 'Pass' if pred == 1 else 'Fail',
        'pass_prob':  float(prob[1]),
        'is_anomaly': bool(np.max(np.abs(X_sc)) > 5.0),
    }

    if model_name == 'Logistic Regression' and hasattr(m, 'coef_'):
        coef, intercept = m.coef_[0], m.intercept_[0]
        z = X_sc[0]; c = coef * z
        resp['logit_decomposition'] = {
            'intercept':     float(intercept),
            'contributions': {fnames[i]: float(c[i]) for i in range(len(fnames))},
            'logit':         float(intercept + np.sum(c)),
        }

    if comparison_data and X_train is not None:
        try:
            from xai_shap import get_shap_values
            sv, bv = get_shap_values(m, X_sc, X_train, model_name)
            resp['shap_values'] = {fnames[i]: float(sv[i]) for i in range(len(fnames))}
            resp['shap_base']   = float(bv)
        except Exception as exc:
            resp['shap_values'] = None; resp['shap_error'] = str(exc)

    if comparison_data and X_train_raw is not None:
        try:
            from xai_lime import get_lime_explanation
            lr = get_lime_explanation(m, sc, X_raw[0], X_train_raw, fnames, model_name)
            resp['lime_weights'] = {fn: float(w) for fn, w in lr}
        except Exception as exc:
            resp['lime_weights'] = None; resp['lime_error'] = str(exc)

    flip_val = None
    for trial_st in np.arange(1.0, 4.01, 0.1):
        if int(m.predict(sc.transform([[trial_st, abs_count, wb]]))[0]) != pred:
            flip_val = float(round(trial_st, 1))
            break
    resp['counterfactual'] = {
        'feature': fnames[0], 'flip_value': flip_val, 'current_value': float(st)}

    resp['sensitivity'] = {
        'studytime':  [float(m.predict_proba(sc.transform([[v, abs_count, wb]]))[0][1])
                       for v in np.linspace(1, 4, 11)],
        'attendance': [float(m.predict_proba(sc.transform([[st, (100-v)/100*93, wb]]))[0][1])
                       for v in np.linspace(0, 100, 11)],
    }
    return jsonify(resp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
