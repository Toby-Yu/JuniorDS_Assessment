\# Car Review Analysis - ThinkCol Jr DS Interview

This project provides a complete pipeline for **\*\*sentiment analysis\*\*** and **\*\*topic extraction\*\*** on a dataset of car reviews. Two approaches are implemented:

\- **\*\*Machine Learning (ML)\*\***: Logistic Regression + TF‑IDF for sentiment, NMF/TF‑IDF with Coherence for topic extraction.

\- **\*\*Generative AI (LLM)\*\***: Few‑shot prompting via SiliconFlow API (DeepSeek‑V4‑Flash) for sentiment, embedding + clustering + LLM naming for topics.

All outputs are saved under \`output/\` and a comparison of sentiment results is automatically generated.

\## File Structure

JuniorDS_Assessment/  
├── data/  
│ └── train.txt # Dataset  
├── config.py # SiliconFlow API configuration  
├── utils.py # HSR, plotting helpers  
├── common/  
│ ├── **init**.py  
│ ├── data_loader.py  
│ └── preprocessing.py # Text cleaning, lemmatization, stop‑words  
├── ml_approach/  
│ ├── sentiment/  
│ │ ├── feature_engineering.py # TF‑IDF vectorizer  
│ │ ├── ml_sentiment.py # Logistic Regression model  
│ │ ├── train.py # Train & save model  
│ │ └── evaluate.py # Evaluate, save predictions & metrics  
│ └── topics/  
│ ├── ml_topics.py # NMF, coherence, UMAP  
│ └── extract_topics.py # Run NMF for K=5..15, save best topics  
├── llm_approach/  
│ ├── llm_utils.py # Shared API call function  
│ ├── sentiment/  
│ │ ├── llm_sentiment.py # Few‑shot prompt & parser  
│ │ └── run_sentiment.py # Classify full test set  
│ └── topics/  
│ ├── llm_topics.py # Embedding, clustering, LLM naming & summary  
│ └── run_topics.py # Full topic extraction pipeline  
├── comparison/  
│ └── compare_sentiment.py # Load metrics, plot comparison chart  
├── test_siliconflow.py # Test API connectivity (combined prompt)  
├── test_llm_sentiment.py # Test LLM sentiment on 2 samples  
├── test_llm_topics.py # Test LLM topic extraction on 40 reviews  
├── .env # Your API key (not committed)  
├── .gitignore  
├── requirements.txt  
└── README.md

\## Setup

1\. \*\*Clone the repository\*\* (or download and extract the project).

2\. \*\*Create and activate a virtual environment\*\*:

\`\`\`bash

python -m venv venv

source venv/bin/activate # Mac/Linux

venv\\Scripts\\activate # Windows

- **Install dependencies**:

bash

pip install -r requirements.txt

- **Set up the SiliconFlow API key**:
  - Create a .env file in the project root with the following content:

text

SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

- - You can obtain a free key from [SiliconFlow](https://siliconflow.cn/). The model deepseek-ai/DeepSeek-V4-Flash is used.

- **Place the dataset** train.txt inside the data/ folder.

**Running the Pipelines**

**1\. ML Approach - Sentiment**

bash

python -m ml_approach.sentiment.train

python -m ml_approach.sentiment.evaluate

- Trains a Logistic Regression model on TF‑IDF features.
- Evaluates on the full test set (245 reviews).
- Saves predictions.csv, metrics.csv, confusion_matrix.png in output/ml/sentiment/.

**2\. ML Approach - Topic Extraction**

bash

python -m ml_approach.topics.extract_topics

- Fits NMF for K = 5 … 15, computes coherence (c_v).
- Saves coherence scores (coherence_scores.csv, coherence_vs_K.png).
- Saves top 2 topic models (word lists) in output/ml/topics/.
- Optionally generates a UMAP visualization.

**3\. LLM Approach - Sentiment**

bash

python -m llm_approach.sentiment.run_sentiment

- Uses few‑shot prompting to classify all 245 test reviews.
- Saves predictions, metrics, and confusion matrix in output/llm/sentiment/.

**4\. LLM Approach - Topic Extraction**

bash

python -m llm_approach.topics.run_topics

- Embeds reviews with all-MiniLM-L6-v2, fits K‑Means for K=5…15, selects K via Calinski‑Harabasz index.
- For the optimal K, samples up to 20 reviews per cluster and asks the LLM for a topic name and a business‑insight summary.
- Saves CH scores (ch_scores.csv, ch_index_vs_K.png) and the final topics with summaries (llm_topics.txt) in output/llm/topics/.

**5\. Sentiment Comparison**

bash

python -m comparison.compare_sentiment

- Reads metrics.csv from both ML and LLM sentiment folders.
- Produces a side‑by‑side metrics table (comparison_metrics.csv) and a grouped bar chart (sentiment_comparison.png) in output/comparison/.

**Quick API Tests (optional)**

- python test_siliconflow.py - tests combined prompt with 2 reviews.
- python test_llm_sentiment.py - classifies 2 reviews using the LLM.
- python test_llm_topics.py - runs embedding + clustering + naming on the first 40 reviews.

**Evaluation Metrics**

**Sentiment Analysis**

- **Accuracy, Precision, Recall, F1‑score (positive class)**
- **Hermione Syntax Ratio (HSR)**: accuracy on reviews containing syntactic conjunctions (but, although, however).
- Confusion matrices for both ML and LLM approaches.

**Topic Extraction**

- **ML (NMF)**: Topic coherence (c_v) - higher indicates more coherent topics. The optimal K is chosen by peak coherence.
- **LLM (Clustering + LLM Naming)**: Calinski‑Harabasz index - higher indicates better cluster separation. The resulting topic names and summaries are directly human‑readable and actionable.

**Output Files**

All results are stored under output/:

- output/ml/sentiment/ - ML sentiment predictions, metrics, confusion matrix.
- output/ml/topics/ - NMF topics (word lists), coherence plot, UMAP (if available).
- output/llm/sentiment/ - LLM sentiment predictions, metrics, confusion matrix.
- output/llm/topics/ - LLM topic names + business summaries, CH index plot.
- output/comparison/ - Combined metrics table and bar chart.