# sentiment_analysis/ml_sentiment.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils import hermione_syntax_ratio

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, class_names=['Neg', 'Pos']):
    y_pred = model.predict(X_test)
    print("\n--- ML Sentiment Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))
    hsr = hermione_syntax_ratio(y_test, y_pred)
    print(f"Hermione Syntax Ratio (pos): {hsr:.3f}")
    return y_pred, hsr