import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..dataloader.dataset import load_data
from pytorch_tabnet.tab_model import TabNetClassifier


def train_tabnet(config, data_dir, train_idx, valid_idx, model_id, model_path, opt_metric, device='cuda'):
    """
    Train and evaluate the given model on the training and test datasets
    """
    X, y, _ = load_data(data_dir)
    criterion = nn.CrossEntropyLoss() if y.ndim ==1 else nn.BCEWithLogitsLoss()

    # Create a TabNetClassifier
    model = TabNetClassifier(
        n_d=config["n_a"],
        n_a=config["n_a"],
        n_steps=config["n_steps"],
        gamma=config["gamma"],
        lambda_sparse=config['lambda_sparse'],
        optimizer_fn = optim.Adam,
        optimizer_params={
            "lr": config["lr"],
        },
        verbose=0,
        seed=config["seed"],
        device_name=device,
    )

    # Train and evaluate the model
    model.fit(X_train=X[train_idx], 
              y_train=y[train_idx],
              eval_set=[(X[valid_idx], y[valid_idx])],
              max_epochs=config["num_epochs"],
              batch_size=config["batch_size"],
              eval_metric=["logloss"],
              patience=config["patience"],
              num_workers=8,
              loss_fn=criterion,
              drop_last=True,
              )
    
    os.makedirs(model_path, exist_ok=True)
    model.save_model(f'{model_path}/tabnet_model')
    
    return model.best_cost

def test_tabnet(model, testset, device='cuda'):
    # print('Evaluating the model...')
    all_probs = model.predict_proba(testset.X)
    all_preds = np.argmax(all_probs, axis=1)
    return all_probs, all_preds