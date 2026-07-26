# llm_approach/sentiment/run_sentiment.py
import sys, os, time, random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from common.data_loader import load_data
from common.preprocessing import add_clean_column
from sklearn.model_selection import train_test_split
from llm_approach.sentiment.llm_sentiment import classify_single_review
from utils import compute_hsr, plot_confusion_matrix

OUTPUT_SENT = os.path.join("output", "llm", "sentiment")

def main():
    os.makedirs(OUTPUT_SENT, exist_ok=True)
    print("Loading data...")
    df = load_data(config.TRAIN_FILE)
    df = add_clean_column(df)
    df['label'] = df['class'].map({'Pos': 1, 'Neg': 0})

    _, X_test, _, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # ---- FULL TEST SET (same 245 reviews as ML) ----
    print(f"Evaluating LLM on {len(X_test)} reviews (full test set)...")
    results = []
    t_start = time.time()
    for idx, (i, text) in enumerate(X_test.items()):
        t0 = time.time()
        predicted = classify_single_review(text)
        elapsed = time.time() - t0
        pred_numeric = 1 if predicted == 'positive' else 0
        results.append({
            'review_index': i,
            'review_text': text,
            'true_label': y_test.loc[i],
            'predicted_label': pred_numeric,
            'predicted_sentiment': predicted
        })
        # Print progress every 10 reviews
        if (idx+1) % 10 == 0 or (idx+1) == len(X_test):
            print(f"  {idx+1}/{len(X_test)}  (last took {elapsed:.2f}s, total {time.time()-t_start:.1f}s)")
        time.sleep(0.3)   # rate limiting

    total_time = time.time() - t_start
    print(f"LLM sentiment completed in {total_time:.1f}s (~{total_time/len(X_test):.1f}s per review)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_SENT, "predictions.csv"), index=False)

    y_true = results_df['true_label']
    y_pred = results_df['predicted_label']

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    hsr = compute_hsr(y_true.values, y_pred.values, results_df['review_text'].values)

    print("\n--- LLM Few‑Shot Sentiment Report ---")
    print(classification_report(y_true, y_pred, target_names=['Neg', 'Pos']))
    print(f"HSR: {hsr:.3f}")

    pd.DataFrame({
        'accuracy': [acc], 'precision': [prec], 'recall': [rec],
        'f1_pos': [f1], 'hsr': [hsr]
    }).to_csv(os.path.join(OUTPUT_SENT, "metrics.csv"), index=False)

    plot_confusion_matrix(y_true, y_pred, labels=[0,1],
                          save_path=os.path.join(OUTPUT_SENT, "confusion_matrix.png"),
                          title="LLM Sentiment Confusion Matrix")

if __name__ == "__main__":
    main()