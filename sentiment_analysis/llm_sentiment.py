# sentiment_analysis/llm_sentiment.py
import requests
import time
from sklearn.metrics import classification_report
from utils import hermione_syntax_ratio

def query_llm_sentiment(text, api_url, headers, max_retries=3):
    prompt = f"""Classify the sentiment of the following car review as only one word: "positive", "negative", or "neutral".

Review: "{text}"

Sentiment:"""

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 10, "temperature": 0.1, "return_full_text": False}
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0]['generated_text'].strip().lower()
            else:
                generated = result.get('generated_text', '').strip().lower()

            if 'positive' in generated:
                return 1
            elif 'negative' in generated:
                return 0
            elif 'neutral' in generated:
                return 2
            else:
                return 0
        except Exception as e:
            print(f"LLM API error: {e}, attempt {attempt+1}")
            time.sleep(2)
    return 0

def evaluate_llm_sentiment(test_texts, y_true, api_url, headers, subset_size=100):
    import random
    indices = list(range(len(test_texts)))
    if len(test_texts) > subset_size:
        indices = random.sample(indices, subset_size)
    else:
        indices = list(range(len(test_texts)))

    print(f"\nEvaluating LLM on {len(indices)} reviews...")
    y_pred = []
    for i in indices:
        pred = query_llm_sentiment(test_texts.iloc[i], api_url, headers)
        y_pred.append(pred)
        if len(y_pred) % 10 == 0:
            print(f"  Progress: {len(y_pred)}/{len(indices)}")

    y_pred_bin = [0 if p == 2 else p for p in y_pred]
    y_true_sub = [y_true.iloc[i] for i in indices]

    print("\n--- LLM Sentiment Classification Report ---")
    print(classification_report(y_true_sub, y_pred_bin, target_names=['Neg', 'Pos']))
    hsr = hermione_syntax_ratio(y_true_sub, y_pred_bin)
    print(f"Hermione Syntax Ratio (pos): {hsr:.3f}")
    return y_true_sub, y_pred_bin, hsr