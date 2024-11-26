import torch
import torch.nn as nn
from torch.optim import Adam
from utils.early_stopping import EarlyStoppingCriterion

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience):
    model.to(device)
    earlystopping = EarlyStoppingCriterion(patience = patience)
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
        
        train_accuracy = total_correct / len(train_loader.dataset)
        val_accuracy, val_loss = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {train_accuracy:.4f}, Val_Loss: {val_loss:.4f}, Val_Accuracy: {val_accuracy:.4f}")

        earlystopping(val_loss, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break
    
    return model

def evaluate_model(model, data_loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

    return (total_correct / len(data_loader.dataset), total_loss)
