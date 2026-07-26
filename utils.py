# utils.py
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def compute_hsr(y_true, y_pred, review_texts):
    """
    Hermione Syntax Ratio: accuracy on reviews containing
    opposing syntactic structures (but, although, however).
    """
    conjunctions = r'\b(but|although|however)\b'
    mask = [bool(re.search(conjunctions, text)) for text in review_texts]
    if sum(mask) == 0:
        return 0.0
    correct = sum((y_true[i] == y_pred[i]) for i in range(len(y_true)) if mask[i])
    return correct / sum(mask)

def plot_comparison(ml_metrics, llm_metrics, metric_names, save_path='comparison.png'):
    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, ml_metrics, width, label='ML (Logistic Regression)')
    ax.bar(x + width/2, llm_metrics, width, label='LLM (DeepSeek-V4-Flash)')
    ax.set_ylabel('Score')
    ax.set_title('Sentiment Analysis Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()