import os
from dotenv import load_dotenv
load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    print("ERROR: SILICONFLOW_API_KEY not set. Add it to .env")
    exit(1)

SILICONFLOW_URL = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

TRAIN_FILE = "data/train.txt"