# run_all.py
import sys
import os

# Load .env early
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment_analysis.run_sentiment import main as run_sentiment
from topic_extraction.run_topics import main as run_topics

if __name__ == "__main__":
    print("="*60)
    print(" STARTING FULL PIPELINE: SENTIMENT + TOPICS")
    print("="*60)

    run_sentiment()

    print("\n" + "="*60)
    print(" SENTIMENT ANALYSIS COMPLETE. MOVING TO TOPIC EXTRACTION...")
    print("="*60)

    run_topics()

    print("\n" + "="*60)
    print(" FULL PIPELINE FINISHED")
    print("="*60)