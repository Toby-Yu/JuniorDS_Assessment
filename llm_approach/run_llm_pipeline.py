# llm_approach/run_llm_pipeline.py
import sys, os, time, random, json, re
import pandas as pd
import requests
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from common.data_loader import load_data
from common.preprocessing import add_clean_column
from sklearn.model_selection import train_test_split
from sentiment.llm_sentiment import parse_sentiment_response
from topics.llm_topics import parse_topics_response
from utils import compute_hsr, plot_confusion_matrix

OUTPUT_SENT = os.path.join("output", "llm", "sentiment")
OUTPUT_TOP = os.path.join("output", "llm", "topics")

def build_combined_prompt(reviews, num_topics=6):
    indexed_reviews = "\n".join([f"{i}: {r[:300]}" for i, r in enumerate(reviews)])
    prompt = f"""
You are a data analyst. Perform the following two tasks on the car reviews provided below.

**Task 1 – Sentiment Classification:**
For each review (identified by its index), classify its sentiment as exactly one of "positive", "negative", or "neutral".
Return the results as a JSON array of objects, each with "index" (integer) and "sentiment" (string).

**Task 2 – Topic Extraction:**
After reading all reviews, identify {num_topics} main topics that customers discuss.
For each topic, provide a short label and a few key words.
Return the topics as a JSON array of strings, each formatted as "Label: keyword1, keyword2, ...".

Output ONLY a valid JSON object with two keys:
- "sentiments": the array of sentiment objects
- "topics": the array of topic strings

Reviews:
{indexed_reviews}
"""
    return prompt

def call_siliconflow(prompt_text, max_tokens=1200):
    payload = {
        "model": config.SILICONFLOW_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise assistant. Always output exactly what is requested in valid JSON."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens
    }
    headers = {"Authorization": f"Bearer {config.SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(config.SILICONFLOW_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 429:
                print("Rate limited, waiting 10s...")
                time.sleep(10)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"API attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return None

def main():
    os.makedirs(OUTPUT_SENT, exist_ok=True)
    os.makedirs(OUTPUT_TOP, exist_ok=True)

    df = load_data(config.TRAIN_FILE)
    df = add_clean_column(df)
    df['label'] = df['class'].map({'Pos': 1, 'Neg': 0})

    _, X_test, _, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    SUBSET_SIZE = 30
    test_indices = X_test.index.tolist()
    llm_idx = random.sample(test_indices, min(SUBSET_SIZE, len(test_indices)))
    llm_texts = X_test.loc[llm_idx].tolist()
    llm_true = y_test.loc[llm_idx]

    combined_prompt = build_combined_prompt(llm_texts, num_topics=6)

    print("Calling SiliconFlow with a combined prompt...")
    response_text = call_siliconflow(combined_prompt, max_tokens=1200)
    if response_text is None:
        print("API call failed.")
        return

    # Parse JSON
    try:
        clean = re.sub(r'```(?:json)?\s*|\s*```', '', response_text).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        json_str = match.group(0) if match else clean
        result = json.loads(json_str)
        sentiments_data = result['sentiments']
        topics_data = result['topics']
    except Exception as e:
        print(f"Failed to parse combined JSON: {e}")
        print("Raw response:", response_text)
        return

    y_pred_raw = parse_sentiment_response(json.dumps(sentiments_data), len(llm_texts))
    # Map neutral (2) -> negative (0) for binary evaluation
    y_pred_bin = [0 if p == 2 else p for p in y_pred_raw]

    detailed = pd.DataFrame({
        'review_index': llm_idx,
        'review_text': llm_texts,
        'true_label': llm_true,
        'predicted_label': y_pred_bin   # 0/1
    })
    detailed.to_csv(os.path.join(OUTPUT_SENT, "predictions.csv"), index=False)

    acc = accuracy_score(llm_true, y_pred_bin)
    f1 = f1_score(llm_true, y_pred_bin, pos_label=1)
    prec = precision_score(llm_true, y_pred_bin, pos_label=1)
    rec = recall_score(llm_true, y_pred_bin, pos_label=1)
    hsr = compute_hsr(llm_true.values, y_pred_bin, llm_texts)

    print("\n--- LLM Sentiment Report ---")
    print(classification_report(llm_true, y_pred_bin, target_names=['Neg', 'Pos']))
    print(f"HSR: {hsr:.3f}")

    pd.DataFrame({
        'accuracy': [acc], 'precision': [prec], 'recall': [rec],
        'f1_pos': [f1], 'hsr': [hsr]
    }).to_csv(os.path.join(OUTPUT_SENT, "metrics.csv"), index=False)

    plot_confusion_matrix(llm_true, y_pred_bin, labels=[0,1],
                          save_path=os.path.join(OUTPUT_SENT, "confusion_matrix.png"),
                          title="LLM Sentiment Confusion Matrix")

    # Topics
    llm_topics_str = parse_topics_response(json.dumps(topics_data))
    with open(os.path.join(OUTPUT_TOP, "llm_topics.txt"), 'w', encoding='utf-8') as f:
        f.write(llm_topics_str)
    with open(os.path.join(OUTPUT_TOP, "llm_topics_raw.txt"), 'w', encoding='utf-8') as f:
        f.write(response_text)

    print("LLM pipeline complete.")

if __name__ == "__main__":
    main()