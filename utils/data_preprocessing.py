import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

nltk.download('punkt_tab')

def clean_text(text):
    # Basic text cleaning (e.g., lowercasing, removing punctuation)
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

def tokenize(text, vocab_size=20000):
    tokens = word_tokenize(text)
    return [hash(token) % vocab_size for token in tokens]

def pad_sequences(sequences, max_len, padding_value=0):
    return [seq[:max_len] + [padding_value] * (max_len - len(seq)) for seq in sequences]

def preprocess_and_split(data_path, test_size=0.2, vocab_size=20000, max_len=100):
    print("Preprocessing Data")
    data = pd.read_csv(data_path)
    data['review'] = data['text'].apply(clean_text)
    data['tokens'] = data['text'].apply(lambda x: tokenize(x, vocab_size))
    data['tokens'] = pad_sequences(data['tokens'], max_len)
    
    X = np.array(data['tokens'].tolist())
    # data['label'] = data['sentiment'].replace({'positive': 0, 'neutral': 1, 'negative': 2})
    y = data['label'].values
    return train_test_split(X, y, test_size=test_size, random_state=42)
