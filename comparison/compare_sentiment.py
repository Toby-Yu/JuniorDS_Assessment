# comparison/compare_sentiment.py
import sys, os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import plot_comparison

ML_METRICS = os.path.join("output", "ml", "sentiment", "metrics.csv")
LLM_METRICS = os.path.join("output", "llm", "sentiment", "metrics.csv")
OUTPUT_DIR = os.path.join("output", "comparison")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ml = pd.read_csv(ML_METRICS).iloc[0]   # single row
    llm = pd.read_csv(LLM_METRICS).iloc[0]

    metrics = ['accuracy', 'precision', 'recall', 'f1_pos', 'hsr']
    ml_vals = [ml[m] for m in metrics]
    llm_vals = [llm[m] for m in metrics]

    # Save combined table
    comp_df = pd.DataFrame({
        'Metric': metrics,
        'ML': ml_vals,
        'LLM': llm_vals
    })
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "comparison_metrics.csv"), index=False)

    # Plot bar chart
    plot_comparison(
        ml_metrics=ml_vals,
        llm_metrics=llm_vals,
        metric_names=metrics,
        save_path=os.path.join(OUTPUT_DIR, "sentiment_comparison.png")
    )
    print("Sentiment comparison saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()