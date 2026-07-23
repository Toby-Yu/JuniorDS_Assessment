# topic_extraction/run_topics.py
import sys
import os

# Load .env early
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from common.data_loader import load_data
from common.preprocessing import add_clean_column
from topic_extraction.ml_topics import run_lda, compute_coherence
from topic_extraction.llm_topics import extract_topics_llm

def main():
    # Load & clean data
    df = load_data(config.TRAIN_FILE)
    df = add_clean_column(df)

    # --- ML Topics (LDA) ---
    print("\n=== ML TOPICS (LDA) ===")
    lda_model, lda_vec = run_lda(df['clean_text'], n_topics=5)
    compute_coherence(lda_model, df['clean_text'], lda_vec)

    # --- LLM Topics ---
    print("\n=== LLM TOPICS ===")
    API_HEADERS = {"Authorization": f"Bearer {config.HF_TOKEN}"}
    sample_texts = df['clean_text'].sample(n=30, random_state=42).tolist()
    llm_topics = extract_topics_llm(sample_texts, config.API_URL, API_HEADERS, num_topics=6)
    if llm_topics:
        print("LLM extracted topics:")
        print(llm_topics)
    else:
        print("Failed to extract topics via LLM.")

if __name__ == "__main__":
    main()