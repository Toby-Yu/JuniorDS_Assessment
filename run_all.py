# run_all.py
import sys
import os

# Ensure the project root is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main functions from the two task runners
from sentiment_analysis.run_sentiment import main as run_sentiment
from topic_extraction.run_topics import main as run_topics

if __name__ == "__main__":
    print("="*60)
    print(" STARTING FULL PIPELINE: SENTIMENT + TOPICS")
    print("="*60)

    # Task 1: Sentiment Analysis
    run_sentiment()

    print("\n" + "="*60)
    print(" SENTIMENT ANALYSIS COMPLETE. MOVING TO TOPIC EXTRACTION...")
    print("="*60)

    # Task 2: Topic Extraction
    run_topics()

    print("\n" + "="*60)
    print(" FULL PIPELINE FINISHED")
    print("="*60)