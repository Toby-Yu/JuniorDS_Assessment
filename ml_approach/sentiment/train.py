# ml_approach/sentiment/train.py
import sys, os, pickle

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import config
from common.data_loader import load_data
from common.preprocessing import add_clean_column
from sklearn.model_selection import train_test_split

# Use relative imports (since we're running as a module)
from .feature_engineering import build_tfidf
from .ml_sentiment import train_model

OUTPUT_DIR = os.path.join("output", "ml", "sentiment")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(config.TRAIN_FILE)
    df = add_clean_column(df)
    df['label'] = df['class'].map({'Pos': 1, 'Neg': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf(X_train, X_test)
    model = train_model(X_train_tfidf, y_train)

    # Save model, vectorizer, and test data for later evaluation
    with open(os.path.join(OUTPUT_DIR, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(OUTPUT_DIR, "vectorizer.pkl"), 'wb') as f:
        pickle.dump(vectorizer, f)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

    print(f"Model and test data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()