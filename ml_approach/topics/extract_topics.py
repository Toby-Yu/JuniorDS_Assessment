# ml_approach/topics/extract_topics.py
import sys, os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from common.data_loader import load_data
from common.preprocessing import add_clean_column
from .ml_topics import (compute_coherence_values, plot_coherence,
                        save_topics, plot_umap)

OUTPUT_DIR = os.path.join("output", "ml", "topics")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading and cleaning data...")
    df = load_data(config.TRAIN_FILE)
    df = add_clean_column(df)

    K_range = list(range(5, 16))
    texts = df['clean_text'].tolist()

    print("Running NMF with varying K...")
    Ks, coh_scores, models, vectorizer, feature_names = compute_coherence_values(
        texts, K_range, max_features=1000
    )

    # ---- Save coherence scores to CSV ----
    coh_df = pd.DataFrame({'K': Ks, 'Coherence': coh_scores})
    csv_path = os.path.join(OUTPUT_DIR, "coherence_scores.csv")
    coh_df.to_csv(csv_path, index=False)
    print(f"Coherence scores saved to {csv_path}")

    # ---- Plot coherence vs K ----
    print("\nGenerating coherence plot...")
    plot_coherence(Ks, coh_scores, save_path=os.path.join(OUTPUT_DIR, "coherence_vs_K.png"))

    # ---- Save top 2 topic sets ----
    print("\nSaving top 2 topic sets...")
    save_topics(models, feature_names, topn=10, output_dir=OUTPUT_DIR)

    # ---- UMAP visualisation ----
    print("\nBuilding document-topic matrix for UMAP...")
    dtm = vectorizer.transform(texts)
    best_model = sorted(models, key=lambda x: x[2], reverse=True)[0][0]
    plot_umap(dtm, best_model, save_path=os.path.join(OUTPUT_DIR, "umap_doc_topics.png"))

    print("\nTopic extraction complete.")

if __name__ == "__main__":
    main()