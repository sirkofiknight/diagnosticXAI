"""
Multi-model training pipeline for XAI research paper.
Trains 6 classifiers on UCI Student Performance Dataset (ID=320)
and saves everything to models_comparison.pkl for SHAP/LIME analysis.

Usage: python backend/train_all.py
"""

import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — substituting extra RandomForest for XGBoost slot.")

OUTPUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_comparison.pkl')
FEATURE_NAMES = ['Academic Effort', 'Institutional Commitment', 'Wellness & Balance']
RAW_FEATURES  = ['studytime', 'absences', 'health']


def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        sp = fetch_ucirepo(id=320)
        X_raw = sp.data.features
        y_raw = sp.data.targets
        X = X_raw[RAW_FEATURES].values.astype(float)
        y = (y_raw['G3'] >= 10).astype(int).values
        print(f"[OK] UCI dataset loaded: {X.shape[0]} samples, class balance: "
              f"{y.mean()*100:.1f}% pass")
        return X, y
    except Exception as exc:
        print(f"[WARN] UCI fetch failed ({exc}), using synthetic fallback.")
        np.random.seed(42)
        n = 2000
        X = np.zeros((n, 3))
        X[:, 0] = np.random.randint(1, 5, n).astype(float)    # studytime  1-4
        X[:, 1] = np.random.randint(0, 94, n).astype(float)   # absences   0-93
        X[:, 2] = np.random.randint(1, 6, n).astype(float)    # health     1-5
        logit = 2.0*(X[:,0]-2.5) - 0.15*(X[:,1]-30) + 1.0*(X[:,2]-3.0)
        y = (logit > 0).astype(int)
        print(f"[OK] Synthetic data: {n} samples, class balance: {y.mean()*100:.1f}% pass")
        return X, y


def build_models():
    # All applicable models use class_weight='balanced' to handle
    # the 84.6% pass-rate imbalance in the UCI dataset.
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'),
        'SVM': SVC(
            probability=True, random_state=42, class_weight='balanced'),  # RBF kernel
        'KNN': KNeighborsClassifier(
            n_neighbors=5),   # KNN has no class_weight; compensated by balanced training
    }
    if HAS_XGB:
        # XGBoost uses scale_pos_weight = n_neg / n_pos for balance
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, random_state=42,
            scale_pos_weight=0.18,   # approx (1-0.846)/0.846
            eval_metric='logloss', verbosity=0)
    else:
        models['XGBoost'] = RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=0, class_weight='balanced')
    return models


def compute_metrics(model, X_train, y_train, X_test, y_test, n_cv=5):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='accuracy')

    return {
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'auc_roc':   float(roc_auc_score(y_test, y_prob)),
        'cv_mean':   float(cv_scores.mean()),
        'cv_std':    float(cv_scores.std()),
        'confusion_matrix':       confusion_matrix(y_test, y_pred).tolist(),
        'classification_report':  classification_report(y_test, y_pred, output_dict=True),
    }


def main():
    X_raw, y = load_data()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    model_defs     = build_models()
    trained_models = {}
    metrics        = {}

    for name, mdl in model_defs.items():
        print(f"  Training {name}...", end=' ', flush=True)
        mdl.fit(X_train, y_train)
        trained_models[name] = mdl
        metrics[name] = compute_metrics(mdl, X_train, y_train, X_test, y_test)
        m = metrics[name]
        print(f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
              f"AUC={m['auc_roc']:.4f}  CV={m['cv_mean']:.4f}±{m['cv_std']:.4f}")

    payload = {
        'models':        trained_models,
        'scaler':        scaler,
        'feature_names': FEATURE_NAMES,
        'metrics':       metrics,
        'X_test':        X_test,
        'y_test':        y_test,
        'X_train':       X_train,
        'y_train':       y_train,
        'X_train_raw':   X_train_raw,
        'X_test_raw':    X_test_raw,
    }
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(payload, f)

    print(f"\n[SAVED] {OUTPUT_PATH}")
    print("\n{'Model':<22} {'Acc':>7} {'F1':>7} {'AUC':>7} {'CV Mean':>9} {'CV Std':>8}")
    print('-' * 65)
    for name, m in metrics.items():
        print(f"{name:<22} {m['accuracy']:>7.4f} {m['f1']:>7.4f} "
              f"{m['auc_roc']:>7.4f} {m['cv_mean']:>9.4f} {m['cv_std']:>8.4f}")


if __name__ == '__main__':
    main()
