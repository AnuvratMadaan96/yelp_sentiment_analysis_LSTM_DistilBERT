import argparse
import yaml
import os
import torch
from scripts.train_lstm import train_lstm_model, evaluate_lstm_model
# from scripts.train_distilbert import train_distilbert_model
from utils.data_preprocessing import preprocess_and_split
from utils.data_loader import create_dataloader
from utils.utils import my_print
from models.lstm_model import LSTMModel

# Device configuration
device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
my_print(f"Device Used: {device}")

def main(args):
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Load configuration
    if args.model == "lstm":
        config_path = os.path.join(project_root, "config", "lstm_config.yaml")
    elif args.model == "distilbert":
        config_path = os.path.join(project_root, "config", "distilbert_config.yaml")
    else:
        raise ValueError("Invalid model type. Choose 'lstm' or 'distilbert'.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Paths for data
    raw_data_path = os.path.join(project_root, "data", "raw", "yelp_reviews.csv")

    # Preprocess data
    X_train, X_val, y_train, y_val = preprocess_and_split(
        raw_data_path,
        test_size=config['test_size'],
        vocab_size=config.get('vocab_size', None),  # Only LSTM needs vocab_size
        max_len=config['max_len']
    )

    # Create DataLoader instances
    train_loader = create_dataloader(X_train, y_train, config['batch_size'], shuffle=True)
    val_loader = create_dataloader(X_val, y_val, config['batch_size'], shuffle=False)

    # Train and Evaluate Model
    if args.model == "lstm":
        my_print("Configuring LSTM Model and Starting to train")
        model = LSTMModel(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            n_layers=config['n_layers'],
            dropout=config['dropout']
        )
        train_lstm_model(config, device, train_loader, val_loader)
    elif args.model == "distilbert":
        # train_distilbert_model(train_loader, val_loader, config, device)
        pass

    # Evaluate Model (optional)
    if args.evaluate:
        model_checkpoint_path = os.path.join(project_root, args.model_checkpoint)
        model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True))
        accuracy = evaluate_lstm_model(model, val_loader, device)
        print(f"Validation Accuracy for {args.model}: {accuracy[0]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yelp Review Sentiment Analysis")
    parser.add_argument("--model", type=str, choices=["lstm", "distilbert"], required=True,
                        help="Specify the model to train: 'lstm' or 'distilbert'")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the trained model on the validation set")
    parser.add_argument("--model-checkpoint", type=str,
                        help="Path to the model checkpoint for evaluation (only required with --evaluate)")
    args = parser.parse_args()

    main(args)