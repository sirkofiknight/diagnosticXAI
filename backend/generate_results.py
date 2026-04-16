"""
Standalone results generator for research paper.
Loads models_comparison.pkl and produces:
  1. LaTeX table printed to stdout
  2. results/model_comparison.png   — grouped bar chart (Acc, F1, AUC)
  3. results/cm_{model}.png         — confusion matrix per model
  4. results/shap_summary_{model}.png — mean |SHAP| bar chart per model

Run from the project root:
    python backend/generate_results.py
"""

import os
import sys
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH    = os.path.join(BACKEND_DIR, 'models_comparison.pkl')
OUT_DIR     = os.path.join(BACKEND_DIR, '..', 'results')
sys.path.insert(0, BACKEND_DIR)


# ── Load data ─────────────────────────────────────────────────────────────────

def load():
    if not os.path.exists(PKL_PATH):
        sys.exit(f"[ERROR] {PKL_PATH} not found. Run python backend/train_all.py first.")
    with open(PKL_PATH, 'rb') as f:
        return pickle.load(f)


# ── 1. LaTeX table ────────────────────────────────────────────────────────────

def print_latex(metrics):
    print("\n% ===== MODEL COMPARISON TABLE =====")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Comparative Performance of Six Classifiers on the UCI Student "
          r"Performance Dataset (ID: 320). CV = 5-fold stratified cross-validation.}")
    print(r"\label{tab:model_comparison}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\hline")
    print(r"\textbf{Model} & \textbf{Accuracy} & \textbf{F1 (macro)} "
          r"& \textbf{AUC-ROC} & \textbf{CV Mean} & \textbf{CV Std} \\")
    print(r"\hline")
    for name, m in metrics.items():
        esc = name.replace('&', r'\&')
        print(f"{esc:<22} & {m['accuracy']:.4f} & {m['f1']:.4f} "
              f"& {m['auc_roc']:.4f} & {m['cv_mean']:.4f} & {m['cv_std']:.4f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# ── 2. Model comparison bar chart ─────────────────────────────────────────────

def plot_model_comparison(metrics, out_dir):
    names = list(metrics.keys())
    accs  = [metrics[n]['accuracy'] for n in names]
    f1s   = [metrics[n]['f1']       for n in names]
    aucs  = [metrics[n]['auc_roc']  for n in names]

    x, w = np.arange(len(names)), 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w,   accs, w, label='Accuracy',    color='#4e79a7', edgecolor='white', linewidth=0.5)
    ax.bar(x,       f1s,  w, label='F1 (macro)',  color='#f28e2b', edgecolor='white', linewidth=0.5)
    ax.bar(x + w,   aucs, w, label='AUC-ROC',     color='#59a14f', edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Multi-Model Performance Comparison — UCI Student Performance Dataset',
                 fontsize=11, pad=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    path = os.path.join(out_dir, 'model_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {path}")


# ── 3. Confusion matrices ─────────────────────────────────────────────────────

def plot_confusion_matrices(metrics, out_dir):
    for name, m in metrics.items():
        cm = np.array(m['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(4, 3.8))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Fail', 'Pass'], fontsize=10)
        ax.set_yticklabels(['Fail', 'Pass'], fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=10, pad=8)
        for i in range(2):
            for j in range(2):
                col = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color=col, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        slug = name.replace(' ', '_').lower()
        path = os.path.join(out_dir, f'cm_{slug}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] {path}")


# ── 4. SHAP summary bar charts ────────────────────────────────────────────────

def plot_shap_summaries(models, X_test, feature_names, out_dir):
    try:
        from xai_shap import get_shap_summary_data
    except ImportError:
        print("[SKIP] xai_shap not importable — skipping SHAP plots.")
        return

    colors = ['#4e79a7', '#f28e2b', '#e15759']
    for name, mdl in models.items():
        print(f"  Computing SHAP summary: {name}...", end=' ', flush=True)
        result    = get_shap_summary_data(mdl, X_test, name)
        mean_abs  = result['mean_abs_shap']

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        y_pos = np.arange(len(feature_names))
        bars  = ax.barh(y_pos, mean_abs, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=9)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=9)
        ax.set_title(f'SHAP Feature Importance — {name}', fontsize=10, pad=8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for bar, val in zip(bars, mean_abs):
            ax.text(bar.get_width() + max(mean_abs)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)
        plt.tight_layout()
        slug = name.replace(' ', '_').lower()
        path = os.path.join(out_dir, f'shap_summary_{slug}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"saved.")

        # Print mean_abs for reference
        for fn, v in zip(feature_names, mean_abs):
            print(f"      {fn}: {v:.5f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = load()
    os.makedirs(OUT_DIR, exist_ok=True)

    models       = data['models']
    metrics      = data['metrics']
    X_test       = data['X_test']
    feature_names = data['feature_names']

    print_latex(metrics)
    plot_model_comparison(metrics, OUT_DIR)
    plot_confusion_matrices(metrics, OUT_DIR)
    plot_shap_summaries(models, X_test, feature_names, OUT_DIR)

    print(f"\n[DONE] All results in: {os.path.abspath(OUT_DIR)}")


if __name__ == '__main__':
    main()
