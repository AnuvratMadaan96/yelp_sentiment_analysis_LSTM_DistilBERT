import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import style
style.use('dark_background')
import torch
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
from transformers import pipeline

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay

def train_distilbert_model(file_path, epochs, train_batch_size, eval_batch_size, device):

    raw_dataset = load_dataset('csv', data_files = file_path)
    data = raw_dataset['train'].train_test_split(test_size=0.1, seed=42)

    checkpoint ="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_func(batch):
        return tokenizer(batch['text'], truncation = True)

    tokenized_datasets = data.map(tokenize_func, batched = True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels =3,
        ignore_mismatched_sizes=True
        )
    
    training_args = TrainingArguments(
        output_dir= 'training_dir',
        evaluation_strategy='epoch',  # means if we want to evaluate model on validation set
        save_strategy='epoch',        # means save model after every epoch which is not a good idea because you will runout of colab space
        num_train_epochs= epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size= eval_batch_size
        )
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset = tokenized_datasets['train'],
        eval_dataset = tokenized_datasets['test'],
        tokenizer = tokenizer,
        )
    
    print("Training Start")
    trainer.train()

    training_dir = os.path.join(os.path.dirname(file_path), "..", "..", "training_dir")
    checkpoint_files = [f for f in os.listdir(training_dir) if f.startswith("checkpoint")]

    if checkpoint_files:
        # Sort files by modified time, newest first
        checkpoint_files = sorted(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(training_dir, x)), reverse=True)
        latest_checkpoint = os.path.join(training_dir, checkpoint_files[0])
        print(f"Latest checkpoint file: {latest_checkpoint}")
    else:
        print("No checkpoint files found in the training directory.")

    savemodel = pipeline('text-classification', model = latest_checkpoint, device = device)

    y_pred = savemodel(data['test']['text'], truncation=True)

    ypred = []
    for i in y_pred:
        if i['label'] == 'LABEL_1':
            ypred.append(1)
        elif i['label'] == 'LABEL_2':
            ypred.append(2)
        else:
            ypred.append(0)
    
    font = {
        'weight' : 'bold',
        'size'   : 15
        }
    
    plt.rc('font', **font)

    cm = confusion_matrix(data['test']['label'], ypred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [0, 1, 2])
    disp.plot(xticks_rotation=45)
    fig = disp.ax_.get_figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.show()

    Accuracy = accuracy_score(data['test']['label'], ypred, normalize=True)
    Precision = precision_score(data['test']['label'], ypred, average = 'macro').round(3)   # macro calculate the average value of classes.
    Recall = recall_score(data['test']['label'], ypred, average = 'macro').round(3)
    F1_Score = f1_score(data['test']['label'], ypred, average = 'macro').round(3)

    print('Accuracy : ', Accuracy)
    print(" ")
    print('Precision : ', Precision)
    print(" ")
    print('Recall : ', Recall)
    print(" ")
    print('F1_Score : ', F1_Score)
    print(" ")
