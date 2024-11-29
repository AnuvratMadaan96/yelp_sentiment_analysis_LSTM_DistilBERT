import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_model import LSTMModel, evaluate_lstm_model

from scripts.train_lstm import train_lstm_model
from utils.data_loader import create_dataloader
from utils.data_preprocessing import preprocess_and_split

device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device Used: {device}")


import os
import yaml

print("Start ----------------->")
# Objective function for Optuna optimization
def objective(trial):
    # Load configuration
    project_root = os.path.dirname(os.path.abspath(__file__))
    # config_path = os.path.join(project_root, "config", "lstm_config.yaml")
    config_path = r"C:\Users\ajayp\Desktop\AJAY\AI with ML Course\Sem-2\Advance Deep Learning\final_project\yelp_sentiment_analysis_LSTM_DistilBERT\config\lstm_config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Hyperparameters to tune
    config['embedding_dim'] = trial.suggest_int('embedding_dim', 50, 300)
    config['hidden_dim'] = trial.suggest_int('hidden_dim', 64, 512)
    config['n_layers'] = trial.suggest_int('n_layers', 1, 3)
    config['dropout'] = trial.suggest_uniform('dropout', 0.1, 0.5)
    config['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Paths for data
    # raw_data_path = os.path.join(project_root, "data", "raw", "yelp_reviews.csv")
    raw_data_path = r"C:\Users\ajayp\Desktop\AJAY\AI with ML Course\Sem-2\Advance Deep Learning\final_project\yelp_sentiment_analysis_LSTM_DistilBERT\data\raw\yelp_reviews.csv"
    
    # Preprocess data
    X_train, X_val, y_train, y_val = preprocess_and_split(
        raw_data_path,
        test_size=config['test_size'],
        vocab_size=config['vocab_size'],
        max_len=config['max_len']
    )

    # Create DataLoader instances
    # train_loader = create_dataloader(X_train, y_train, config['batch_size'], shuffle=True) og
    # val_loader = create_dataloader(X_val, y_val, config['batch_size'], shuffle=False) og

    print(type(X_train), type(y_train))
    print("-----------")

    X_train = torch.tensor(X_train) if not isinstance(X_train, torch.Tensor) else X_train
    y_train = torch.tensor(y_train) if not isinstance(y_train, torch.Tensor) else y_train

    X_val = torch.tensor(X_val) if not isinstance(X_val, torch.Tensor) else X_val
    y_val = torch.tensor(y_val) if not isinstance(y_val, torch.Tensor) else y_val


    print(type(X_train), type(y_train))
    print("-----------")
    # Assuming X_train, y_train, X_val, and y_val are PyTorch tensors
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Assuming batch_size is defined in the config
    batch_size = config['batch_size']

    # Define DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize and train the LSTM model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = LSTMModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        n_layers=config['n_layers'],
        dropout=config['dropout']
    )
    model.to(device)

    # Train the model and get validation accuracy
    train_lstm_model(config, device, train_loader, val_loader)
    accuracy, _ = evaluate_lstm_model(model, val_loader, device)
    
    return accuracy

# Optimize hyperparameters
def optimize_hyperparameters():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best accuracy: {study.best_value}")

if __name__ == "__main__":
    import optuna
    import torch
    from scripts.train_lstm import train_lstm_model
    from utils.data_loader import create_dataloader
    from utils.data_preprocessing import preprocess_and_split
    from models.lstm_model import LSTMModel

    import os
    import yaml
    optimize_hyperparameters()