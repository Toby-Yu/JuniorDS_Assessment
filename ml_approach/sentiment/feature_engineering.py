# ml_approach/sentiment/feature_engineering.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from common.preprocessing import CAR_STOP_WORDS_SENTIMENT

def build_tfidf(train_texts, test_texts, max_features=5000):
    stop_words = list(sk_text.ENGLISH_STOP_WORDS.union(CAR_STOP_WORDS_SENTIMENT))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words=stop_words
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer