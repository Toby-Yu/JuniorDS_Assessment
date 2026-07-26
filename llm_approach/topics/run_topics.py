# llm_approach/topics/run_topics.py
import sys, os, time
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from common.data_loader import load_data
from common.preprocessing import add_clean_column
from llm_approach.topics.llm_topics import (
    embed_texts, compute_ch_scores, select_best_k, cluster_texts,
    sample_reviews_per_cluster, process_clusters, get_fallback_count
)

OUTPUT_TOP = os.path.join("output", "llm", "topics")

def main():
    os.makedirs(OUTPUT_TOP, exist_ok=True)
    t_total = time.time()

    print("Loading data...")
    df = load_data(config.TRAIN_FILE)
    df = add_clean_column(df)
    texts = df['clean_text'].tolist()
    print(f"Corpus size: {len(texts)} reviews")

    print("\n--- Step 1/4: Embedding ---")
    t0 = time.time()
    embeddings = embed_texts(texts)
    print(f"Embedding completed in {time.time()-t0:.1f}s")

    K_range = list(range(5, 16))
    print(f"\n--- Step 2/4: Computing K-Means for K={K_range[0]}...{K_range[-1]} ---")
    t0 = time.time()
    ch_scores, models = compute_ch_scores(embeddings, K_range)
    print(f"CH scores computed in {time.time()-t0:.1f}s")
    for k, s in zip(K_range, ch_scores):
        print(f"  K={k}: CH={s:.2f}")

    ch_df = pd.DataFrame({'K': K_range, 'Calinski_Harabasz': ch_scores})
    ch_df.to_csv(os.path.join(OUTPUT_TOP, "ch_scores.csv"), index=False)

    plt.figure(figsize=(8,4))
    plt.plot(K_range, ch_scores, marker='o')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Calinski-Harabasz Score")
    plt.title("Calinski-Harabasz Index vs K")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_TOP, "ch_index_vs_K.png"))
    plt.show()

    best_k, best_idx, best_score = select_best_k(ch_scores, K_range)
    print(f"Selected best K = {best_k} (CH = {best_score:.2f})")

    print(f"\n--- Step 3/4: Clustering with K={best_k} ---")
    t0 = time.time()
    labels, km = cluster_texts(embeddings, best_k)
    print(f"Clustering completed in {time.time()-t0:.1f}s")

    print("\n--- Step 4/4: Sampling reviews, generating topic names & summaries ---")
    t0 = time.time()
    sampled = sample_reviews_per_cluster(texts, labels, max_per_cluster=20)
    print("Sampled cluster sizes:")
    for lab, revs in sampled.items():
        print(f"  Cluster {lab}: {len(revs)} reviews")

    print("\nGenerating topic names and summaries via LLM (resilient)...")
    topic_names, summaries = process_clusters(sampled)
    print(f"\nLLM processing completed in {time.time()-t0:.1f}s")

    fallbacks = get_fallback_count()
    if fallbacks > 0:
        print(f"\n*** WARNING: {fallbacks} cluster(s) used local keyword fallback. ***")
    else:
        print("\n*** All topic names and summaries successfully generated via LLM. ***")

    # Save results
    with open(os.path.join(OUTPUT_TOP, "llm_topics.txt"), 'w', encoding='utf-8') as f:
        for lab in sorted(topic_names.keys()):
            f.write(f"Cluster {lab}: {topic_names[lab]}\n")
            f.write(f"Summary: {summaries[lab]}\n\n")
    print(f"\nTopic names and summaries saved to {OUTPUT_TOP}/llm_topics.txt")

    print(f"\nTotal topic extraction time: {time.time()-t_total:.1f}s")
    print("Topic extraction complete.")

if __name__ == "__main__":
    main()