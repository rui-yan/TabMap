import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ..dataloader.dataset import load_data
from ..dataloader.data_loader import TabularDataset


class CNN_2d(nn.Module):
    def __init__(self, input_dim, output_dim, num_fc1=100, kernel_size=3, dropout=0.1):
        super(CNN_2d, self).__init__()

        Cin, Hin, Win = 1, input_dim[0], input_dim[1]
        init_f = 8
        
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=kernel_size, padding=1)
        self.conv1_bn = nn.BatchNorm2d(init_f)
        h, w=findConv2dOutShape(Hin, Win, self.conv1, pool=0)
        
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=kernel_size, padding=1)
        self.conv2_bn = nn.BatchNorm2d(2*init_f)
        h, w=findConv2dOutShape(h, w, self.conv2, pool=0)        
        
        self.num_flatten = h * w * 2 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc1_bn = nn.BatchNorm1d(num_fc1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_fc1, output_dim)
    
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = nn.functional.relu(self.conv1_bn(x))
        x = self.conv2(x)
        x = nn.functional.relu(self.conv2_bn(x))
        x = x.contiguous().view(-1, self.num_flatten)
        x = self.fc1(x)
        x = nn.functional.relu(self.fc1_bn(x))
        
        if return_features:
            return x
        
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_2DCNN(config, data_dir, train_idx, valid_idx, model_id, model_path, opt_metric, device='cuda'):
    """
    Train and validate a 2D CNN model on the training and validation datasets
    """
    if config["seed"] is not None:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
    
    _, y, _ = load_data(data_dir)
    X_img = np.load(f'{data_dir}/{model_id}.npy')
    
    train_dataset = TabularDataset(X_img[train_idx], y[train_idx])
    valid_dataset = TabularDataset(X_img[valid_idx], y[valid_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=int(config["batch_size"]), num_workers=8, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    model = CNN_2d(input_dim=train_dataset.input_size, 
                   output_dim=train_dataset.n_classes,
                   num_fc1=config["num_fc1"], 
                   kernel_size=config["kernel_size"], 
                   dropout=config["dropout"])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Train the model
    if opt_metric == 'balanced_accuracy':
        best_val_balanced_accuracy = 0
    elif opt_metric == 'loss':
        best_val_loss = float('inf')

    counter = 0  # Counter for early stopping
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
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
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
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
                torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pth'))
            else:
                counter += 1
        elif opt_metric == 'loss':
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pth'))
            else:
                counter += 1
                
        if counter >= config["patience"]:
            print(f'\nEarly stopping triggered')
            break
    
    if opt_metric == 'balanced_accuracy':
        return best_val_balanced_accuracy
    elif opt_metric == 'loss':
        return best_val_loss


def test_2DCNN(model, testset, device='cuda'):
    """
    Evaluate a 2D CNN model on the test dataset.
    """
    # print('Evaluating the model...')
    test_loader = DataLoader(testset, shuffle=False)
    model.eval()
    model.to(device)
    
    all_probs, all_preds = [], []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    return all_probs, all_preds

def findConv2dOutShape(hin, win, conv, pool=2):
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation
    
    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    
    return int(hout),int(wout)