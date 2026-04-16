"""
LIME integration for multi-model XAI analysis.
Works with raw (unscaled) feature values; the predict_fn handles scaling internally.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


def get_lime_explanation(model, scaler, X_instance_unscaled, X_train_unscaled,
                         feature_names, model_name):
    """
    Compute LIME explanation for a single (unscaled) instance.

    The explainer is fitted on X_train_unscaled; the predict_fn wraps
    scaler.transform → model.predict_proba so LIME perturbs in the
    interpretable raw-feature space.

    Parameters
    ----------
    model                : fitted sklearn-compatible classifier
    scaler               : fitted StandardScaler
    X_instance_unscaled  : array-like shape (n_features,) — raw feature values
    X_train_unscaled     : array-like shape (n_train, n_features) — raw training data
    feature_names        : list[str]
    model_name           : str (used only for logging)

    Returns
    -------
    list of (feature_name, weight) tuples sorted by abs(weight) descending
    """
    if not HAS_LIME:
        return [(fn, 0.0) for fn in feature_names]

    X_train_raw = np.array(X_train_unscaled)
    X_inst      = np.array(X_instance_unscaled).flatten()

    def predict_fn(X):
        return model.predict_proba(scaler.transform(X))

    try:
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train_raw,
            feature_names=feature_names,
            class_names=['Fail', 'Pass'],
            mode='classification',
            discretize_continuous=True,
            random_state=42,
        )
        exp = explainer.explain_instance(
            X_inst,
            predict_fn,
            num_features=len(feature_names),
        )

        # as_map() returns {class_idx: [(feature_idx, weight), ...]}
        # feature_idx corresponds to the column index in training_data
        exp_map = exp.as_map()
        weights = {fn: 0.0 for fn in feature_names}
        if 1 in exp_map:
            for feat_idx, weight in exp_map[1]:
                if feat_idx < len(feature_names):
                    weights[feature_names[feat_idx]] = float(weight)

        return sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)

    except Exception as exc:
        print(f"[LIME warn] {model_name}: {exc}")
        return [(fn, 0.0) for fn in feature_names]
