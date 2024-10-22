import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ..dataloader.dataset import load_data
from ..dataloader.data_loader import TabularDataset
from tab_transformer_pytorch import TabTransformer


def train_TabTransformer(config,  data_dir, train_idx, valid_idx, model_id, model_path, opt_metric, device='cuda'):
    """
    Train and evaluate the given model on the training and test datasets
    """
    X, y, _ = load_data(data_dir)
    
    train_dataset = TabularDataset(X[train_idx], y[train_idx])
    valid_dataset = TabularDataset(X[valid_idx], y[valid_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=int(config["batch_size"]), num_workers=8, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    model = TabTransformer(categories=(),
                           num_continuous=train_dataset.input_size, 
                           dim_out=train_dataset.n_classes, 
                           mlp_hidden_mults=(4, 2),
                           dim=32,
                           mlp_act=nn.ReLU(),
                           depth=config["depth"],
                           heads=config["heads"],
                           attn_dropout=config["attn_dropout"],
                           ff_dropout=config["ff_dropout"])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Train the model
    if opt_metric == 'balanced_accuracy':
        best_val_balanced_accuracy = 0.0
    elif opt_metric == 'loss':
        best_val_loss = float('inf')
    
    counter = 0  # Counter for early stopping
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            cat = torch.empty(0).to(device)
            cont = inputs.to(device)
            labels = labels.to(device)
                
            optimizer.zero_grad()
            outputs = model(cat, cont)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                cat = torch.empty(0).to(device)
                cont = inputs.to(device)
                labels = labels.to(device)

                outputs = model(cat, cont)
                loss = criterion(outputs, labels.float() if labels.ndim > 1 else labels)
                val_loss += loss.item()
                
                pred = outputs.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        val_loss /= len(valid_loader.dataset)
        val_balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        print(f"\rEpoch {epoch+1}/{config['num_epochs']}: Train Loss: {train_loss:.4f}, " +
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_balanced_accuracy:.4f}", end='')
        
        os.makedirs(model_path, exist_ok=True)
        if opt_metric == 'balanced_accuracy':
            if val_balanced_accuracy > best_val_balanced_accuracy:
                best_val_balanced_accuracy = val_balanced_accuracy
                counter = 0
                torch.save(model.state_dict(), f'{model_path}/best_model.pth')
            else:
                counter += 1
        elif opt_metric == 'loss':
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), f'{model_path}/best_model.pth')
            else:
                counter += 1
        
        if counter >= config["patience"]:
            print(f'\nEarly stopping triggered')
            break
    
    if opt_metric == 'balanced_accuracy':
        return best_val_balanced_accuracy
    elif opt_metric == 'loss':
        return best_val_loss


def test_TabTransformer(model, testset, device='cuda'):
    # print('Evaluating the model...')
    test_loader = DataLoader(testset, shuffle=False)
    model.eval()
    model = model.to(device)
    
    all_probs, all_preds = [], []
    with torch.no_grad():
        for inputs, _ in test_loader:
            cat = torch.empty(0).to(device)
            cont = inputs.to(device)
            outputs = model(cat, cont)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    return all_probs, all_preds