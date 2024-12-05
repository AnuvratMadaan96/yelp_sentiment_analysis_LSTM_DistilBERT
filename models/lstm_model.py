import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output

def evaluate_lstm_model(model, data_loader, device):
    """
    Evaluates the LSTM model on the provided data loader.
    
    Parameters:
        model (torch.nn.Module): The trained LSTM model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation or test dataset.
        device (torch.device): Device to perform computation (CPU or CUDA).
        
    Returns:
        float: Accuracy of the model on the given dataset.
        dict: Additional metrics, such as loss or other evaluation scores (if needed).
    """
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predicted classes
            _, predicted = torch.max(outputs, 1)
            
            # Collect predictions and true labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return accuracy, {"all_predictions": all_predictions, "all_labels": all_labels}
