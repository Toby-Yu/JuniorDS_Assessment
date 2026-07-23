# Car Review Analysis – ThinkCol Jr DS Interview

## Setup

1. Clone the repository.
2. Create a virtual environment:  
   `python -m venv venv`  
   `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
3. Install dependencies:  
   `pip install -r requirements.txt`
4. Create a `.env` file in the root and add:  
   `HF_TOKEN=hf_your_actual_token`
5. Place `train.txt` inside the `data/` folder.
6. Run the full pipeline:  
   `python run_all.py`

## Individual tasks

- Sentiment only: `python sentiment_analysis/run_sentiment.py`
- Topics only: `python topic_extraction/run_topics.py`