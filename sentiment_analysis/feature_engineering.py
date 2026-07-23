# sentiment_analysis/feature_engineering.py
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(train_texts, test_texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer