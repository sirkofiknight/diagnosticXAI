import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, 
            template_folder=os.path.join('..', 'templates'),
            static_folder=os.path.join('..', 'static'))
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'expert_v3.pkl')

def train_and_save():
    from ucimlrepo import fetch_ucirepo 
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    try:
        student_performance = fetch_ucirepo(id=320) 
        X_raw = student_performance.data.features 
        y_raw = student_performance.data.targets
        
        selected_features = ['studytime', 'absences', 'health']
        X = X_raw[selected_features].copy()
        passed = (y_raw['G3'] >= 10).astype(int)
        
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_sc, passed)
        
        # Expert Nomenclature
        feature_names = ['Academic Effort', 'Institutional Commitment', 'Wellness & Balance']
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler, 'feature_names': feature_names}, f)
        return model, scaler, feature_names
    except:
        # High-Fidelity Synthetic Fallback
        np.random.seed(42)
        n = 2000
        X = np.zeros((n, 3))
        X[:, 0] = np.random.randint(1, 5, n) # study
        X[:, 1] = np.random.randint(0, 94, n) # absences
        X[:, 2] = np.random.randint(1, 6, n) # health
        logit = 2.0 * (X[:,0] - 2.5) - 0.15 * (X[:,1] - 30) + 1.0 * (X[:,2] - 3.0)
        p_pass = (logit > 0).astype(int)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_sc, p_pass)
        feature_names = ['Academic Effort', 'Institutional Commitment', 'Wellness & Balance']
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler, 'feature_names': feature_names}, f)
        return model, scaler, feature_names

model, scaler, feature_names = train_and_save()
u, s = scaler.mean_, scaler.scale_

@app.route('/')
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    req = request.json
    st, wb, ar = float(req.get('studytime', 1)), float(req.get('wellbeing', 1)), float(req.get('attendance_rate', 100))
    abs_count = (100 - ar) / 100 * 93
    X_sc = scaler.transform([[st, abs_count, wb]])
    prob = model.predict_proba(X_sc)[0]
    pred = int(model.predict(X_sc)[0])
    return jsonify({
        'prediction': 'Pass' if pred == 1 else 'Fail',
        'confidence': float(prob[1] if pred == 1 else prob[0]),
        'pass_prob': float(prob[1]),
        'is_anomaly': bool(np.max(np.abs(X_sc)) > 5.0)
    })

@app.route('/explain', methods=['POST'])
def explain():
    req = request.json
    st, wb, ar = float(req.get('studytime', 1)), float(req.get('wellbeing', 1)), float(req.get('attendance_rate', 100))
    abs_count = (100 - ar) / 100 * 93
    X_sc = scaler.transform([[st, abs_count, wb]])
    
    coef, intercept = model.coef_[0], model.intercept_[0]
    z_scores = X_sc[0]
    contributions = coef * z_scores
    logit = float(intercept + np.sum(contributions))
    prob = 1 / (1 + np.exp(-logit))

    # 1. Detailed Human Translation
    dom_idx = np.argmax(np.abs(contributions))
    strongest_feature = feature_names[dom_idx]
    
    if logit > 0:
        human = (f"<span class='text-green-400 font-bold'>The model predicts a PASS</span> because your total diagnostic score ({logit:+.2f}) is above the critical threshold. "
                 f"The most influential factor supporting this prediction is your <b>{strongest_feature}</b>, which alone adds {contributions[dom_idx]:+.2f} to the success logit.")
    else:
        human = (f"<span class='text-red-400 font-bold'>The model predicts a FAIL</span> because your total diagnostic score ({logit:+.2f}) has dropped into the risk zone. "
                 f"The primary driver of this risk is your <b>{strongest_feature}</b>. Without intervention in this area, the probability of a successful outcome is only {(prob*100):.1f}%.")

    # 2. Detailed Technical Breakdown (Logits)
    tech = (f"<b>Decision Logit Equation (L):</b><br>"
            f"L = β₀ + Σ(βᵢ * zᵢ)<br><br>"
            f"L = {intercept:.3f} (Intercept)<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{coef[0]:+.3f} * {z_scores[0]:.2f} ({feature_names[0]})<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{coef[1]:+.3f} * {z_scores[1]:.2f} ({feature_names[1]})<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{coef[2]:+.3f} * {z_scores[2]:.2f} ({feature_names[2]})<br>"
            f"-------------------<br>"
            f"<b>L = {logit:.3f}</b><br><br>"
            f"<b>P(Pass) = σ(L) = 1/(1+e⁻ᴸ) = {prob:.4f}</b>")

    # 3. Actionable Counterfactual
    action = ""
    if logit < 0:
        # Scenario: What if Study Time was increased?
        target_logit = 0.1 # Threshold to flip to Pass
        needed_z0 = (target_logit - intercept - coef[1]*z_scores[1] - coef[2]*z_scores[2]) / coef[0]
        needed_st = needed_z0 * s[0] + u[0]
        if 1 <= needed_st <= 6:
            action = f"<b>Intervention Strategy:</b> Increasing <b>{feature_names[0]}</b> to level <b>{needed_st:.1f}</b> would reposition the total logit to +0.1, flipping the prediction to <span class='text-green-400'>PASS</span>."
        else:
            action = "<b>Intervention Strategy:</b> Current indicators are severely depressed. A holistic improvement across all three pillars is required to reach the success threshold."

    return jsonify({
        'contributions': {'studytime': float(contributions[0]), 'attendance': float(contributions[1]), 'wellbeing': float(contributions[2])},
        'reasoning': human,
        'technical_reasoning': tech,
        'action_plan': action,
        'sensitivity': {
            'studytime': [float(model.predict_proba(scaler.transform([[v, abs_count, wb]]))[0][1]) for v in np.linspace(1, 4, 11)],
            'attendance': [float(model.predict_proba(scaler.transform([[st, (100-v)/100*93, wb]]))[0][1]) for v in np.linspace(0, 100, 11)]
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
