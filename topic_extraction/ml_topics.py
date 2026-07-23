# topic_extraction/ml_topics.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def run_lda(texts, n_topics=5, max_features=1000):
    vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words='english', max_features=max_features)
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online')
    lda.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n--- LDA Topics (n={n_topics}) ---")
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        print(f"Topic {idx}: {' | '.join(top_words)}")
    return lda, vectorizer

def compute_coherence(lda_model, texts, vectorizer):
    try:
        from gensim.models.coherencemodel import CoherenceModel
        from gensim.corpora.dictionary import Dictionary
        tokenized = [doc.split() for doc in texts]
        dictionary = Dictionary(tokenized)
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
        cm = CoherenceModel(model=lda_model, texts=tokenized, dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
        print(f"LDA Coherence (c_v): {coherence:.3f}")
        return coherence
    except ImportError:
        print("gensim not installed – skipping coherence score.")
        return None