# common/preprocessing.py
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# ----- Car-specific stop words for SENTIMENT (light removal) -----
CAR_STOP_WORDS_SENTIMENT = {
    'br', 'quot', 'car', 'ford', 'vehicle', 'just', 'get', 'got', 'would',
    'one', 'also', 'much', 'even', 'make', 'drive', 'driving',
    'driven', 'drives', 'many', 'though', 'still', 'back', 'front', 'first', 'lt'
}

# ----- Car-specific stop words for TOPIC EXTRACTION (aggressive removal) -----
CAR_STOP_WORDS_TOPICS = CAR_STOP_WORDS_SENTIMENT.union({
    'like', 'well', 'really', 'new', 'year', 'years', 'two', '000', 'mile', 'miles',
    'vehicle', 'truck', '150', 'van', 'seat', 'seats', 'car', 'ford',
    'time', 'problem', 'problems', 'dealer', 'day', 'days', 'week', 'months',
    'great', 'good', 'bad', 'nice', 'best', 'worst', 'little', 'sure'
})

def clean_text(text, lemmatize=True):
    text = str(text).lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if lemmatize:
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(w) for w in words]
        text = ' '.join(lemmatized)
    return text

def add_clean_column(df, col='text', new_col='clean_text'):
    df[new_col] = df[col].apply(clean_text)
    return df