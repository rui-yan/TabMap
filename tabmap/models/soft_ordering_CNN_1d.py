import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ..dataloader.dataset import load_data
from ..dataloader.data_loader import TabularDataset


class SoftOrdering1DCNN(nn.Module):
    def __init__(self, input_dim, output_dim, sign_size=32, cha_input=16, cha_hidden=32, 
                 K=2, dropout_input=0.2, dropout_output=0.2, dropout_hidden=0.2):
        super(SoftOrdering1DCNN, self).__init__()
        
        hidden_size = sign_size * cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size // 2
        output_size = (sign_size // 4) * cha_hidden
        
        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = nn.Conv1d(
            cha_input, 
            cha_input*K, 
            kernel_size=5, 
            stride=1, 
            padding=2,
            groups=cha_input, 
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)
        
        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            groups=cha_hidden, 
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)
        
        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)
        
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))
        
        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x =  x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x


def train_1DCNN(config, data_dir, train_idx, valid_idx, model_id, model_path, opt_metric, device='cuda'):
    """
    Train and evaluate a 1D CNN model on the training and test datasets
    """
    if config["seed"] is not None:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

    X, y, _ = load_data(data_dir)
    
    train_dataset = TabularDataset(X[train_idx], y[train_idx])
    valid_dataset = TabularDataset(X[valid_idx], y[valid_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=int(config["batch_size"]), num_workers=8, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    model = SoftOrdering1DCNN(input_dim=train_dataset.input_size, 
                              output_dim=train_dataset.n_classes,
                              dropout_input=config['dropout_input'],
                              dropout_output=config['dropout_output'],
                              dropout_hidden=config['dropout_hidden'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Train the model
    if opt_metric == 'balanced_accuracy':
        best_val_balanced_accuracy = 0.0
    elif opt_metric == 'loss':
        best_val_loss = float('inf')
    
    counter = 0 # Counter for early stopping
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

        # Validation loss
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
    

def test_1DCNN(model, testset, device='cuda'):
    """
    Evaluate a 1D CNN model on the test dataset.
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