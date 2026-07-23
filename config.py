# config.py
import os

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN environment variable not set. LLM calls will fail.")
    print("Set it via .env file or export HF_TOKEN='your_token'")

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
TRAIN_FILE = "data/train.txt"