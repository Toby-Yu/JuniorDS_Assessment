# sentiment_analysis/run_sentiment.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import config
from common.data_loader import load_data
from common.preprocessing import add_clean_column
from sentiment_analysis.feature_engineering import build_tfidf
from sentiment_analysis.ml_sentiment import train_model, evaluate_model
from sentiment_analysis.llm_sentiment import evaluate_llm_sentiment
from utils import plot_comparison

def main():
    # Load & preprocess
    df = load_data(config.TRAIN_FILE)
    df = add_clean_column(df)
    df['label'] = df['class'].map({'Pos': 1, 'Neg': 0})

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # --- ML Sentiment ---
    print("\n=== ML SENTIMENT ===")
    X_train_tfidf, X_test_tfidf, _ = build_tfidf(X_train, X_test)
    ml_model = train_model(X_train_tfidf, y_train)
    y_pred_ml, ml_hsr = evaluate_model(ml_model, X_test_tfidf, y_test)
    ml_acc = accuracy_score(y_test, y_pred_ml)
    ml_f1 = f1_score(y_test, y_pred_ml, pos_label=1)

    # --- LLM Sentiment ---
    print("\n=== LLM SENTIMENT ===")
    API_HEADERS = {"Authorization": f"Bearer {config.HF_TOKEN}"}
    y_true_llm, y_pred_llm_bin, llm_hsr = evaluate_llm_sentiment(
        X_test, y_test, config.API_URL, API_HEADERS, subset_size=50
    )
    llm_acc = accuracy_score(y_true_llm, y_pred_llm_bin)
    llm_f1 = f1_score(y_true_llm, y_pred_llm_bin, pos_label=1)

    # --- Comparison ---
    print("\n=== SENTIMENT COMPARISON ===")
    print(f"ML  – Accuracy: {ml_acc:.3f}, F1: {ml_f1:.3f}, HSR: {ml_hsr:.3f}")
    print(f"LLM – Accuracy: {llm_acc:.3f}, F1: {llm_f1:.3f}, HSR: {llm_hsr:.3f}")

    plot_comparison(
        ml_metrics=[ml_acc, ml_f1, ml_hsr],
        llm_metrics=[llm_acc, llm_f1, llm_hsr],
        metric_names=['Accuracy', 'F1 (pos)', 'HSR (pos)'],
        save_path='sentiment_comparison.png'
    )

if __name__ == "__main__":
    main()