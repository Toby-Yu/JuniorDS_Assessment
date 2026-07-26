# test_siliconflow.py
import requests, json, os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not API_KEY:
    print("Please set SILICONFLOW_API_KEY in .env")
    exit(1)

URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"

# Two example reviews
reviews = [
    "I absolutely love my new Ford. The seats are comfortable and it drives smoothly.",
    "The transmission failed after 30,000 miles. Very disappointed."
]

# ------------------------------------------------------------------
# Build the combined prompt (same as in run_llm_pipeline.py)
# ------------------------------------------------------------------
indexed_reviews = "\n".join([f"{i}: {r[:300]}" for i, r in enumerate(reviews)])
num_topics = 3

combined_prompt = f"""
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

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a precise assistant. Always output exactly what is requested in valid JSON."},
        {"role": "user", "content": combined_prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 600
}

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

print("Testing combined sentiment + topic prompt...")
resp = requests.post(URL, headers=headers, json=payload)
print("Status:", resp.status_code)

if resp.status_code == 200:
    content = resp.json()['choices'][0]['message']['content']
    print("\nRaw response:")
    print(content)

    # Try to parse the JSON
    import re
    try:
        clean = re.sub(r'```(?:json)?\s*|\s*```', '', content).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        json_str = match.group(0) if match else clean
        result = json.loads(json_str)

        print("\n--- Parsed Sentiments ---")
        for s in result['sentiments']:
            print(f"  Review {s['index']}: {s['sentiment']}")

        print("\n--- Parsed Topics ---")
        for t in result['topics']:
            print(f"  - {t}")
    except Exception as e:
        print("Failed to parse JSON:", e)
        print("Check the raw response format.")
else:
    print("Error:", resp.text)