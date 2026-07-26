# test_llm_sentiment.py
import time
from llm_approach.sentiment.llm_sentiment import classify_single_review

reviews = [
    "I absolutely love my new Ford. The seats are comfortable and it drives smoothly.",
    "The transmission failed after 30,000 miles. Very disappointed."
]

t0 = time.time()
for i, r in enumerate(reviews):
    t1 = time.time()
    pred = classify_single_review(r)
    elapsed = time.time() - t1
    print(f"Review {i}: {pred}  (took {elapsed:.2f}s)")
print(f"Total time: {time.time()-t0:.2f}s")