# topic_extraction/llm_topics.py
import requests

def extract_topics_llm(review_texts, api_url, headers, num_topics=6):
    truncated = [t[:300] for t in review_texts]
    combined = "\n".join([f"- {t}" for t in truncated])
    prompt = f"""You are a data analyst. Below are several car reviews. Identify {num_topics} main topics that customers discuss. For each topic, provide a short label and a few key words. Output the topics in a numbered list.

Reviews:
{combined}

Topics:"""

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.3, "return_full_text": False}
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list):
            return result[0]['generated_text'].strip()
        else:
            return result.get('generated_text', '').strip()
    except Exception as e:
        print(f"LLM topic extraction error: {e}")
        return None