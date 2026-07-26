# test_llm_topics.py
import sys, os, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_FILE
from common.data_loader import load_data
from common.preprocessing import add_clean_column
from llm_approach.topics.llm_topics import (
    embed_texts, compute_ch_scores, select_best_k, cluster_texts,
    sample_reviews_per_cluster, process_clusters, get_fallback_count
)

df = load_data(TRAIN_FILE)
df = add_clean_column(df)
texts = df['clean_text'].tolist()[:40]
print(f"Testing with {len(texts)} reviews.")

t_total = time.time()

t0 = time.time()
emb = embed_texts(texts)
print(f"Embedding took {time.time()-t0:.2f}s")

K_range = [2,3,4,5]
print("Computing CH scores...")
t0 = time.time()
ch_scores, models = compute_ch_scores(emb, K_range)
print(f"CH computation took {time.time()-t0:.2f}s")
for k, s in zip(K_range, ch_scores):
    print(f"  K={k}: CH={s:.2f}")

best_k, idx, best_score = select_best_k(ch_scores, K_range)
print(f"Best K = {best_k} (CH={best_score:.2f})")

t0 = time.time()
labels, km = cluster_texts(emb, best_k)
print(f"Clustering took {time.time()-t0:.2f}s")

t0 = time.time()
sampled = sample_reviews_per_cluster(texts, labels, max_per_cluster=20)
print(f"Sampling took {time.time()-t0:.2f}s")
print("Sampled cluster sizes:", {lab: len(r) for lab, r in sampled.items()})

print("Generating topic names and summaries via LLM...")
t0 = time.time()
topic_names, summaries = process_clusters(sampled)
print(f"LLM processing took {time.time()-t0:.2f}s")
for lab in sorted(topic_names.keys()):
    print(f"  Cluster {lab}: {topic_names[lab]}")
    print(f"    Summary: {summaries[lab]}")

fallbacks = get_fallback_count()
if fallbacks > 0:
    print(f"\n*** WARNING: {fallbacks} cluster(s) used local keyword fallback. ***")
else:
    print("\n*** All topic names and summaries successfully generated via LLM. ***")

print(f"Total test time: {time.time()-t_total:.2f}s")