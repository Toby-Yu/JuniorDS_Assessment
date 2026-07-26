import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['class', 'text']]
    df.dropna(subset=['text'], inplace=True)
    return df