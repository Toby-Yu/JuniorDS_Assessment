# llm_approach/topics/llm_topics.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from llm_approach.llm_utils import call_siliconflow
import random, time

_fallback_count = 0

def embed_texts(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)

def compute_ch_scores(embeddings, K_range):
    scores = []
    models = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        ch = calinski_harabasz_score(embeddings, labels)
        scores.append(ch)
        models.append(km)
    return scores, models

def select_best_k(scores, K_range):
    best_idx = np.argmax(scores)
    return K_range[best_idx], best_idx, scores[best_idx]

def cluster_texts(embeddings, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km

def sample_reviews_per_cluster(texts, labels, max_per_cluster=20):
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(texts[i])
    sampled = {}
    for lab, revs in clusters.items():
        if len(revs) <= max_per_cluster:
            sampled[lab] = revs
        else:
            sampled[lab] = random.sample(revs, max_per_cluster)
    return sampled

def _fallback_topic_name(reviews, top_n=5):
    try:
        vec = TfidfVectorizer(stop_words='english', max_features=top_n)
        vec.fit(reviews)
        top_words = vec.get_feature_names_out()
        return ' '.join(top_words).title()
    except:
        return "Various Car Topics"

def _fallback_summary(reviews, topic_name, top_n=8):
    try:
        vec = TfidfVectorizer(stop_words='english', max_features=top_n)
        vec.fit(reviews)
        top_words = vec.get_feature_names_out()
        return f"Reviews mentioning {', '.join(top_words)}."
    except:
        return "Various car experiences."

def generate_topic_name(reviews, api_caller=call_siliconflow):
    """Generate a topic name with retries and fallback."""
    global _fallback_count
    combined = "\n".join([f"- {r[:200]}" for r in reviews])
    prompt = f"""Below are several car reviews that share a common theme.

{combined}

What single short phrase (3-5 words) best describes the unifying topic of these reviews? Answer ONLY with the phrase, without additional text or quotation marks."""

    for attempt in range(3):
        try:
            response = api_caller(prompt, max_tokens=30, temperature=0.3, timeout=120, max_retries=1)
            if response and len(response.strip()) > 0:
                return response.strip()
            else:
                print(f"Warning: empty response on attempt {attempt+1}")
        except Exception as e:
            print(f"Warning: error on attempt {attempt+1}: {e}")
        time.sleep(2 ** attempt)

    _fallback_count += 1
    print(f"ALERT: All LLM attempts failed for topic naming. Using local keyword fallback. (Total fallbacks: {_fallback_count})")
    return _fallback_topic_name(reviews)

def generate_topic_summary(reviews, topic_name, api_caller=call_siliconflow):
    """
    Generate a business‑insight summary: what are the main issues/praises,
    which car models are mentioned, and what actions should the business consider.
    """
    combined = "\n".join([f"- {r[:200]}" for r in reviews])
    prompt = f"""The following car reviews are all about "{topic_name}".

{combined}

As a data analyst, write a short summary (2-3 sentences) that covers:
- What customers mainly praise or complain about related to this topic.
- Which specific car models (if any) are frequently mentioned.
- One actionable recommendation for the business.

Answer ONLY with the summary, without any additional formatting."""

    for attempt in range(3):
        try:
            response = api_caller(prompt, max_tokens=100, temperature=0.3, timeout=120, max_retries=1)
            if response and len(response.strip()) > 0:
                return response.strip()
            else:
                print(f"Warning: empty summary on attempt {attempt+1}")
        except Exception as e:
            print(f"Warning: error on attempt {attempt+1}: {e}")
        time.sleep(2 ** attempt)

    # Fallback summary
    print("ALERT: All LLM attempts failed for summary. Using local keyword fallback.")
    return _fallback_summary(reviews, topic_name)

def process_clusters(sampled_clusters):
    """
    For each cluster, generate a topic name and a summary.
    Returns:
        topic_names: dict cluster_id -> name
        summaries: dict cluster_id -> summary
    """
    global _fallback_count
    _fallback_count = 0
    topic_names = {}
    summaries = {}
    for lab, revs in sorted(sampled_clusters.items()):
        print(f"\nProcessing cluster {lab} ({len(revs)} reviews)")
        name = generate_topic_name(revs)
        topic_names[lab] = name
        print(f"  Name: {name}")

        summary = generate_topic_summary(revs, name)
        summaries[lab] = summary
        print(f"  Summary: {summary}")
    return topic_names, summaries

def get_fallback_count():
    return _fallback_count