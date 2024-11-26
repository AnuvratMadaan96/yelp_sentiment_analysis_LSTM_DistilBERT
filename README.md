# Yelp Sentiment Analysis

This project aims to perform sentiment analysis on Yelp reviews using two deep learning models: **LSTM** and **DistilBERT**. The models predict the sentiment (positive, neutral, or negative) of reviews based on their text and assigned ratings.

## Project Overview

Objective: Compare the performance, strengths, and weaknesses of **LSTM** and **DistilBERT** models for sentiment analysis on Yelp reviews.
Data: Yelp review dataset containing review texts and their associated ratings.
Labels:
	•	Rating > 3: Positive
	•	Rating < 3: Negative
	•	Rating = 3: Neutral

### File Structure
```
yelp_sentiment_analysis/
├── data/                        # Folder for storing raw and preprocessed data
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Processed datasets (e.g., tokenized, padded)
├── models/                      # Model definitions
│   ├── lstm_model.py            # LSTM model definition
│   ├── distilbert_model.py      # DistilBERT model definition
├── utils/                       # Utility functions
│   ├── data_preprocessing.py    # Text cleaning, tokenization, padding, etc.
│   ├── data_loader.py           # Dataset and DataLoader creation
│   ├── evaluation.py            # Functions for accuracy, F1-score, etc.
│   ├── train_utils.py           # Training and evaluation loops
├── notebooks/                   # Jupyter Notebooks for exploratory analysis
├── scripts/                     # Scripts for training and testing
│   ├── train_lstm.py            # Script to train the LSTM model
│   ├── train_distilbert.py      # Script to train the DistilBERT model
│   ├── test_model.py            # Script to test models
├── config/                      # Configuration files
│   ├── lstm_config.yaml         # Hyperparameters for LSTM
│   ├── distilbert_config.yaml   # Hyperparameters for DistilBERT
├── results/                     # Folder to save results, metrics, and logs
├── checkpoints/                 # Folder to save trained models
├── main.py                      # Entry point for the project
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
```

### Setup Instructions
1. Clone the Repository
```bash
git clone https://github.com/AnuvratMadaan96/yelp_sentiment_analysis_LSTM_DistilBERT.git
cd yelp_sentiment_analysis
```

2. Create a Virtual Environment
``` bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies
``` bash
pip install -r requirements.txt
```

### Usage

#### Train the LSTM Model
``` bash
python main.py --model lstm
```

#### Configuration

Modify the hyperparameters in the YAML files located in the config/ directory:
	•	LSTM: lstm_config.yaml
	•	DistilBERT: distilbert_config.yaml

#### Results

	•	Performance metrics (accuracy, precision, recall, F1-score) for both models will be saved in the results/ folder.
	•	Logs and training progress are saved in the same directory for analysis.

### Contributing
Feel free to contribute to this project by submitting a pull request or reporting issues.
