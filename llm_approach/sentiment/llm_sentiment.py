# llm_approach/sentiment/llm_sentiment.py
import json, re
from llm_approach.llm_utils import call_siliconflow

def few_shot_sentiment_prompt(review_text):
    examples = """
Example 1: "I absolutely love my new Ford. The seats are comfortable and it drives smoothly." -> {"sentiment": "positive"}
Example 2: "The transmission failed after 30,000 miles. Very disappointed." -> {"sentiment": "negative"}
Example 3: "The car is okay, but nothing special." -> {"sentiment": "negative"}
"""
    prompt = f"""Classify the sentiment of the following car review as exactly one of "positive" or "negative". 
Return ONLY a JSON object with the key "sentiment".

{examples}
Review: "{review_text}"
Output:"""
    return prompt


def parse_sentiment_response(response_text):
    """
    Parse the LLM response and return either 'positive' or 'negative'.
    If the response cannot be parsed, a warning is printed and 'negative' is returned
    as a safe default.
    """
    if not response_text:
        print("Warning: empty response from LLM – defaulting to 'negative'")
        return "negative"

    clean = response_text.strip().lower()

    # 1) Try to extract a JSON object with a "sentiment" key
    try:
        match = re.search(r'\{.*"sentiment".*\}', clean, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            sent = obj.get('sentiment', '').lower()
            if sent in ('positive', 'negative'):
                return sent
    except Exception:
        pass

    # 2) Fallback: look for the words "positive" or "negative"
    if 'positive' in clean:
        return 'positive'
    if 'negative' in clean:
        return 'negative'

    # 3) If nothing matched, log the unexpected output and default to negative
    print(f"Warning: unexpected LLM response – defaulting to 'negative'. Response was: '{response_text[:200]}'")
    return "negative"


def classify_single_review(review_text):
    prompt = few_shot_sentiment_prompt(review_text)
    response = call_siliconflow(prompt, max_tokens=30)
    return parse_sentiment_response(response)