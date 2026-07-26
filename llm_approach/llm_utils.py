# llm_approach/llm_utils.py
import requests, time, sys, os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

def call_siliconflow(prompt_text, max_tokens=100, temperature=0.1, timeout=30, max_retries=3):
    payload = {
        "model": config.SILICONFLOW_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise assistant. Always output valid JSON."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {"Authorization": f"Bearer {config.SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(config.SILICONFLOW_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                print("Rate limited, waiting 10s...")
                time.sleep(10)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"API attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    return None