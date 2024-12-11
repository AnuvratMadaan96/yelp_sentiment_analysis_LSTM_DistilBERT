import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from models.lstm_model import LSTMModel
from utils.data_preprocessing import preprocess_and_split
from utils.data_loader import create_dataloader
from utils.train_utils import train_model, evaluate_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
from matplotlib import style
style.use('dark_background')
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

    trained_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)  # Get predicted class indices
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert predictions and true labels to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    
    font = {
        'weight' : 'bold',
        'size'   : 15
        }
    
    plt.rc('font', **font)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [0, 1, 2])
    disp.plot(xticks_rotation=45)
    fig = disp.ax_.get_figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.show()

    Accuracy = accuracy_score(all_labels, all_preds, normalize=True)
    Precision = precision_score(all_labels, all_preds, average = 'macro').round(3)   # macro calculate the average value of classes.
    Recall = recall_score(all_labels, all_preds, average = 'macro').round(3)
    F1_Score = f1_score(all_labels, all_preds, average = 'macro').round(3)

    print('Accuracy : ', Accuracy)
    print(" ")
    print('Precision : ', Precision)
    print(" ")
    print('Recall : ', Recall)
    print(" ")
    print('F1_Score : ', F1_Score)
    print(" ")

def evaluate_lstm_model(model, val_loader, device):

    # Loss and Optimizer
    criterion = CrossEntropyLoss()

    # evaluate the model
    return evaluate_model(model, val_loader, criterion, device)
