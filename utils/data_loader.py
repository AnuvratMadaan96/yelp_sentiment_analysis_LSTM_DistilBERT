import torch
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def create_dataloader(X, y, batch_size, shuffle=True):
    dataset = SentimentDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
