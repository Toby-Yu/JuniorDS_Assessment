# ml_approach/topics/ml_topics.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.decomposition import NMF
from common.preprocessing import CAR_STOP_WORDS_TOPICS
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import time
import os

# ------------------------------------------------------------
# UMAP is imported only when needed, inside plot_umap()
# ------------------------------------------------------------
_umap_available = None

def _check_umap():
    global _umap_available
    if _umap_available is None:
        try:
            import umap
            _umap_available = True
        except (ImportError, ModuleNotFoundError):
            _umap_available = False
            print("UMAP not available – skipping UMAP plot.")
    return _umap_available


def compute_coherence_values(texts, K_range, max_features=1000):
    """
    Loop over K values, fit NMF, compute coherence (c_v).
    Returns: Ks, coherence scores, list of (model, top_words, coherence),
             fitted vectorizer, feature_names.
    """
    stop_words = list(sk_text.ENGLISH_STOP_WORDS.union(CAR_STOP_WORDS_TOPICS))
    vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=5,
        stop_words=stop_words,
        max_features=max_features
    )
    print("Building TF‑IDF matrix...")
    dtm = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Vocabulary size: {len(feature_names)}")

    # Pre‑tokenise for coherence (done once)
    print("Tokenising for coherence...")
    tokenized = [doc.split() for doc in texts]
    dictionary = Dictionary(tokenized)

    coherence_scores = []
    models = []
    for k in K_range:
        print(f"\n--- K = {k} ---")
        t0 = time.time()
        nmf = NMF(n_components=k, random_state=42, max_iter=500)
        nmf.fit(dtm)
        print(f"  NMF fit done in {time.time()-t0:.1f}s")

        # Extract top words for each topic
        topics_words = []
        for topic_idx in range(k):
            top_indices = nmf.components_[topic_idx].argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_indices]
            topics_words.append(top_words)

        # Compute coherence (fast with processes=1)
        t0 = time.time()
        cm = CoherenceModel(
            topics=topics_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence='c_v',
            processes=1          # single process – avoids overhead
        )
        coh = cm.get_coherence()
        coherence_scores.append(coh)
        models.append((nmf, topics_words, coh))
        print(f"  Coherence = {coh:.4f}  (computed in {time.time()-t0:.1f}s)")

    return K_range, coherence_scores, models, vectorizer, feature_names


def plot_coherence(K_range, coherence_scores, save_path="output/ml/topics/coherence_vs_K.png"):
    plt.figure(figsize=(8,4))
    plt.plot(K_range, coherence_scores, marker='o')
    plt.xlabel("Number of Topics (K)")
    plt.ylabel("Coherence (c_v)")
    plt.title("NMF Topic Coherence vs K")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


def save_topics(models, feature_names, topn=10, output_dir="output/ml/topics"):
    os.makedirs(output_dir, exist_ok=True)
    sorted_models = sorted(models, key=lambda x: x[2], reverse=True)
    for rank, (nmf, topics_words, coh) in enumerate(sorted_models[:2]):
        fname = os.path.join(output_dir,
                             f"nmf_topics_rank{rank+1}_K{nmf.n_components}_coherence{coh:.4f}.txt")
        with open(fname, 'w', encoding='utf-8') as f:
            for idx, words in enumerate(topics_words):
                line = f"Topic {idx}: {' | '.join(words[:topn])}"
                f.write(line + "\n")
                print(line)
        print(f"Saved topics to {fname}")


def plot_umap(dtm, nmf_model, save_path="output/ml/topics/umap_doc_topics.png"):
    if not _check_umap():
        return
    # Import umap only here – no top‑level import
    import umap
    W = nmf_model.transform(dtm)
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(W)
    plt.figure(figsize=(8,6))
    plt.scatter(embedding[:,0], embedding[:,1], s=5, alpha=0.5)
    plt.title("UMAP Projection of Document-Topic Matrix (NMF)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()