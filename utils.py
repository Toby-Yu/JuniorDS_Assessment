# utils.py
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

def hermione_syntax_ratio(y_true, y_pred, pos_label=1):
    """Harmonic mean of precision and recall for the positive class."""
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def plot_comparison(ml_metrics, llm_metrics, metric_names, save_path='comparison.png'):
    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, ml_metrics, width, label='ML (Logistic Regression)')
    ax.bar(x + width/2, llm_metrics, width, label='LLM (Mistral 7B)')
    ax.set_ylabel('Score')
    ax.set_title('Sentiment Analysis Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()