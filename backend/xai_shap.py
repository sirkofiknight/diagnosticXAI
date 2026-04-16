"""
SHAP integration for multi-model XAI analysis.
Selects the correct explainer type per model family and normalises
output to a consistent 1-D array for the positive class (Pass=1).
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_pos_class(shap_values):
    """Return 1-D SHAP array for the positive class from any SHAP output format."""
    if isinstance(shap_values, list):
        # sklearn Tree/Kernel for binary: list[neg_class, pos_class]
        arr = np.array(shap_values[1])
    else:
        arr = np.array(shap_values)
    # (n_samples, n_features, n_classes) → pick pos class
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    # (n_samples, n_features) → first sample
    if arr.ndim == 2:
        return arr[0]
    return arr.flatten()


def _base_value(explainer, shap_values):
    """Extract base/expected value for the positive class."""
    ev = getattr(explainer, 'expected_value', 0.0)
    if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
        return float(ev[1])
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[0])
    return float(ev)


# ── Public API ────────────────────────────────────────────────────────────────

def get_shap_values(model, X_instance, X_background, model_name):
    """
    Compute SHAP values for a single (already-scaled) instance.

    Explainer selection:
      LogisticRegression           → LinearExplainer
      Decision Tree, Random Forest → TreeExplainer
      XGBoost                      → TreeExplainer
      SVM, KNN                     → KernelExplainer  (kmeans background, k=50)

    Returns
    -------
    shap_vals : np.ndarray shape (n_features,) — positive-class SHAP values
    base_val  : float — model expected value for positive class
    """
    if not HAS_SHAP:
        n = np.array(X_instance).reshape(1, -1).shape[1]
        return np.zeros(n), 0.0

    X_inst = np.array(X_instance).reshape(1, -1)
    X_bg   = np.array(X_background)

    try:
        if model_name == 'Logistic Regression':
            exp = shap.LinearExplainer(model, X_bg,
                                       feature_perturbation='interventional')
            sv  = exp.shap_values(X_inst)

        elif model_name in ('Decision Tree', 'Random Forest', 'XGBoost'):
            exp = shap.TreeExplainer(model)
            sv  = exp.shap_values(X_inst)

        else:   # SVM, KNN → KernelExplainer
            k   = min(50, len(X_bg))
            bg  = shap.kmeans(X_bg, k)
            exp = shap.KernelExplainer(model.predict_proba, bg)
            sv  = exp.shap_values(X_inst, nsamples=100, silent=True)

        return _to_pos_class(sv), _base_value(exp, sv)

    except Exception as exc:
        print(f"[SHAP warn] {model_name}: {exc}")
        return np.zeros(X_inst.shape[1]), 0.0


def get_shap_summary_data(model, X_test, model_name):
    """
    Compute mean |SHAP| across the test set (for summary bar plots).

    Returns
    -------
    dict with key 'mean_abs_shap': list of floats, one per feature
    """
    if not HAS_SHAP:
        n = np.array(X_test).shape[1] if hasattr(X_test, '__len__') else 3
        return {'mean_abs_shap': [0.0] * n}

    X = np.array(X_test)

    try:
        if model_name == 'Logistic Regression':
            exp = shap.LinearExplainer(model, X)
            sv  = exp.shap_values(X)

        elif model_name in ('Decision Tree', 'Random Forest', 'XGBoost'):
            exp = shap.TreeExplainer(model)
            sv  = exp.shap_values(X)

        else:   # SVM, KNN — use a subset for speed
            n_eval = min(100, len(X))
            k      = min(50, n_eval)
            bg     = shap.kmeans(X[:n_eval], k)
            exp    = shap.KernelExplainer(model.predict_proba, bg)
            sv     = exp.shap_values(X[:n_eval], nsamples=50, silent=True)

        # Normalise to shape (n_samples, n_features) for positive class
        if isinstance(sv, list):
            arr = np.abs(np.array(sv[1]))
        else:
            arr = np.abs(np.array(sv))
            if arr.ndim == 3:
                arr = arr[:, :, 1]

        return {'mean_abs_shap': arr.mean(axis=0).tolist()}

    except Exception as exc:
        print(f"[SHAP summary warn] {model_name}: {exc}")
        return {'mean_abs_shap': [0.0] * X.shape[1]}
