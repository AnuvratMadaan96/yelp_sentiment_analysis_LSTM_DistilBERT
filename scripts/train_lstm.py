import torch
import os
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from models.lstm_model import LSTMModel
from utils.data_preprocessing import preprocess_and_split
from utils.data_loader import create_dataloader
from utils.train_utils import train_model, evaluate_model
import yaml

def train_lstm_model(config, device, train_loader, val_loader):

    # Model Initialization
    model = LSTMModel(
        vocab_size=config['vocab_size'], 
        embedding_dim=config['embedding_dim'], 
        hidden_dim=config['hidden_dim'], 
        output_dim=config['output_dim'], 
        n_layers=config['n_layers'], 
        dropout=config['dropout']
    )

    print(model)

    # Loss and Optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, config['epochs'], config['patience'])

    # Save the trained model
    torch.save(trained_model.state_dict(), "checkpoints/lstm_model.pth")

def evaluate_lstm_model(model, val_loader, device):

    # Loss and Optimizer
    criterion = CrossEntropyLoss()

    # evaluate the model
    return evaluate_model(model, val_loader, criterion, device)
