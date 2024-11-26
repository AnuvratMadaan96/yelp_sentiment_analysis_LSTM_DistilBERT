import torch
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class PrintDecorator:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("***")
        result = self.func(*args, **kwargs)
        print("***")
        return result

# Usage example
@PrintDecorator
def my_print(statement):
    print(statement)

# Tokenizer
def tokenize(text):
    tokens = word_tokenize(text.lower())
    return [hash(token) % 20000 for token in tokens]

# Save Model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        review = self.data.loc[index, 'review']
        label = self.data.loc[index, 'label']
        tokens = self.tokenizer(review)
        tokens = tokens[:self.max_len] + [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# DataLoader
def create_dataloader(data_path, tokenizer, max_len, batch_size, shuffle=True):
    dataset = SentimentDataset(data_path, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)