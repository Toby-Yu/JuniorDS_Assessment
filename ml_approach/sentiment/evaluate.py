# ml_approach/sentiment/evaluate.py
import sys, os, pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import compute_hsr, plot_confusion_matrix

OUTPUT_DIR = os.path.join("output", "ml", "sentiment")

def main():
    with open(os.path.join(OUTPUT_DIR, "model.pkl"), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "vectorizer.pkl"), 'rb') as f:
        vectorizer = pickle.load(f)

    X_test = pd.read_csv(os.path.join(OUTPUT_DIR, "X_test.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(OUTPUT_DIR, "y_test.csv")).squeeze()

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    hsr = compute_hsr(y_test.values, y_pred, X_test.values)   # X_test contains cleaned text

    # Save predictions with review texts for HSR and comparison
    results_df = pd.DataFrame({
        'review_index': range(len(y_test)),
        'review_text': X_test.values,
        'true_label': y_test.values,
        'predicted_label': y_pred
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

    metrics = pd.DataFrame({
        'accuracy': [acc],
        'precision': [prec],
        'recall': [rec],
        'f1_pos': [f1],
        'hsr': [hsr]
    })
    metrics.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

    print(f"ML Sentiment Metrics: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, HSR={hsr:.3f}")

    plot_confusion_matrix(y_test, y_pred, labels=[0,1],
                          save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
                          title="ML Sentiment Confusion Matrix")

if __name__ == "__main__":
    main()