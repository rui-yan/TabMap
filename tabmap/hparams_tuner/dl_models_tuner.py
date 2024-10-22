import logging
import os
import shutil
import optuna
from functools import partial

import torch
import torch.nn as nn
import numpy as np

# Import models
from ..models.CNN_2d import CNN_2d, train_2DCNN
from ..models.soft_ordering_CNN_1d import SoftOrdering1DCNN, train_1DCNN
from ..models.tabnet import train_tabnet
from ..models.tabtransformer import train_TabTransformer
from pytorch_tabnet.tab_model import TabNetClassifier
from tab_transformer_pytorch import TabTransformer

# Import configurations
from ..config import DL_MODELS, DL_MODELS_TRANSFORMER_BASED


class DLModelTuner:
    """
    A class to handle the tuning and training of deep learning models.
    """
    def __init__(
        self,
        data_config,
        train_idx,
        valid_idx,
        model_id,
        fold_id=None,
        results_path=None,
        use_default_hparams=False,
        opt_metric='loss',
        random_seed=None
    ):
        """
        Initializes the DLModelTuner with the provided configurations.

        Parameters:
            data_config (dict): Configuration for the dataset.
            train_idx (np.ndarray): Indices for training data.
            valid_idx (np.ndarray): Indices for validation data.
            model_id (str): Identifier for the model to be used.
            fold_id (str, optional): Identifier for cross-validation fold.
            results_path (str, optional): Path to store results and models.
            use_default_hparams (bool): Flag to use default hyperparameters.
            opt_metric (str): Optimization metric ('loss' or 'balanced_accuracy').
            random_seed (int, optional): Random seed for reproducibility.
        """
        self.data_config = data_config
        self.train_idx, self.valid_idx = train_idx, valid_idx
        self.model_id = model_id
        self.opt_metric = opt_metric
        self.use_default_hparams = use_default_hparams
        self.results_path = results_path or '../results'
        
        if fold_id is not None:
            self.model_fpath = f"{data_config['data_set']}/{data_config['data_set']}_{fold_id}/{model_id}"
        else:
            self.model_fpath = f"{data_config['data_set']}/{data_config['data_set']}/{model_id}"
        
        # Set up environment
        self.seed = random_seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
        
        # Set up model specifics
        self.model, self.train_func = self._get_dl_model()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if use_default_hparams: 
            self.h_parameters = self._hyperparameters()
            self.final_model, self.params_dict = self._train_model_with_default_hparams()
        else:
            self.h_parameters = self._hyperparameters()
            self.cv_results = self._cross_validation()
            self.final_model, self.params_dict = self._best_model()
            self._remove_ckpts()
    
    def _get_dl_model(self):
        """
        Retrieve the deep learning model and training function based on model ID.
        
        Returns:
            tuple: (model_class, training_function)
        """
        model_registry = {
            'TabMap': (CNN_2d, train_2DCNN),
            'DeepInsight': (CNN_2d, train_2DCNN),
            'IGTD': (CNN_2d, train_2DCNN),
            '1DCNN': (SoftOrdering1DCNN, train_1DCNN),
            'TabNet': (TabNetClassifier, train_tabnet),
            'TabTransformer': (TabTransformer, train_TabTransformer),
        }
        if self.model_id in DL_MODELS:
            return model_registry.get(self.model_id, (None, None))
        else:
            raise ValueError(f"Invalid model_id: {self.model_id}")

    def _hyperparameters(self):
        """
        Define hyperparameters for model training.
        
        Returns:
            dict: Hyperparameters for the specified model.
        """
        # Hyperparameter settings for both default and optimized cases
        default_hparams = {
            'TabMap': {"dropout": 0.1, "num_fc1": 100, "kernel_size": 3},
            'DeepInsight': {"dropout": 0.1, "num_fc1": 100, "kernel_size": 3},
            'IGTD': {"dropout": 0.1, "num_fc1": 100, "kernel_size": 3},
            '1DCNN': {"dropout_input": 0.1, "dropout_output": 0.1, "dropout_hidden": 0.1},
            'TabNet': {"n_a": 64, "n_steps": 3, "gamma": 1.3, "lambda_sparse": 0},
            'TabTransformer': {"depth": 6, "heads": 8, "attn_dropout": 0.1, "ff_dropout": 0.15},
        }
        
        # Optimized hyperparameter ranges for tuning
        optimized_hparams = {
            'TabMap': 
                {
                    "dropout": [0.1, 0.3],
                    "num_fc1": [100, 128],
                    "kernel_size": [3, 5],
                },
            'DeepInsight': 
                {
                    "dropout": [0.1, 0.3],
                    "num_fc1": [100, 128],
                    "kernel_size": [3, 5],
                },
            'IGTD': 
                {
                    "dropout": [0.1, 0.3],
                    "num_fc1": [100, 128],
                    "kernel_size": [3, 5],
                },
            '1DCNN': 
                {
                    "dropout_input": [0.0, 0.1, 0.3], 
                    "dropout_output": [0.0, 0.1, 0.3], 
                    "dropout_hidden": [0.0, 0.1, 0.3],
                },
            '2DCNN': 
                {
                    "dropout": [0.1, 0.3], 
                    "num_fc1": [100, 128],
                    "kernel_size": [3, 5],
                },
            'TabNet': 
                {
                    "n_a": [8, 16, 32, 64], 
                    "n_steps": [3, 4, 5], 
                    'lambda_sparse': [0, 0.000001, 0.0001, 0.001, 0.01, 0.1], 
                    "gamma": [1.0, 1.3],
                },
            'TabTransformer': 
                {
                    "depth": [4, 6], 
                    "heads": [6, 8], 
                    "attn_dropout": [0.05, 0.1], 
                    "ff_dropout": [0.1, 0.15]
                },
        }

        if self.use_default_hparams:
            return default_hparams.get(self.model_id, {})
        else:
            return optimized_hparams.get(self.model_id, {})
    
    def objective(self, trial, train_config, h_parameters, train_func, data_dir, train_idx, valid_idx,
                  model_id, model_path, opt_metric, device):
        """
        Objective function for hyperparameter optimization using Optuna.
        
        Parameters:
            trial (optuna.trial.Trial): Optuna trial object.
            train_config (dict): Training configuration parameters.
            h_parameters (dict): Hyperparameter ranges.
            train_func (function): Training function for the model.
            data_dir (str): Directory containing the data.
            train_idx (np.ndarray): Indices for training data.
            valid_idx (np.ndarray): Indices for validation data.
            model_id (str): Identifier for the model.
            model_path (str): Path to save the model.
            opt_metric (str): Metric to optimize ('loss' or 'balanced_accuracy').
            device (str): Device to run computations ('cuda' or 'cpu').

        Returns:
            float: Validation metric to optimize.
        """
        hyperparameter_values = {}
        params_dict = {**train_config, **h_parameters}
        
        for param_name, param_range in params_dict.items():
            if isinstance(param_range, list):
                hyperparameter_values[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                hyperparameter_values[param_name] = param_range[0]
        
        logging.info(f"Trying hyperparameters: {hyperparameter_values}")
        validation_metric = train_func(config=hyperparameter_values,
                data_dir=data_dir,
                train_idx=train_idx, valid_idx=valid_idx,
                model_id=model_id,
                model_path=os.path.join(model_path, f'trial_{trial.number}'),
                opt_metric=opt_metric,
                device=device)

        return validation_metric  # Return the value to be minimized or maximized
    
    def _cross_validation(self):
        """
        Configure and execute a cross-validation study to optimize model hyperparameters.
        
        Returns:
            optuna.study.Study: The study object containing optimization results.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            optuna.samplers.TPESampler.RNG_SEED = self.seed
        
        # Training configuration parameters
        if self.model_id in ['TabTransformer']:
            train_config = {
                "lr": [1e-3],
                "patience": [10],
                "seed": [self.seed],
                "batch_size": [64, 128],
                "num_epochs": [100, 200]
            }
        else:
            train_config = {
                "lr": [1e-3],
                "patience": [10],
                "seed": [self.seed],
                "batch_size": [32, 64],
                "num_epochs": [30, 50]
            }

        direction = "maximize" if self.opt_metric == 'balanced_accuracy' else "minimize"
        cv_results = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=self.seed))
        cv_results.optimize(
            partial(
                self.objective,
                train_config=train_config,
                h_parameters=self.h_parameters,
                train_func=self.train_func,
                data_dir=self.data_config["data_dir"],
                train_idx=self.train_idx, 
                valid_idx=self.valid_idx,
                model_id=self.model_id,
                model_path=os.path.join(self.results_path, self.model_fpath),
                opt_metric=self.opt_metric,
                device=self.device
            ), 
            n_trials=10)
        
        return cv_results
    
    def _train_model_with_default_hparams(self):
        """
        Train the model using default hyperparameters.
        """
        if self.model_id in DL_MODELS_TRANSFORMER_BASED:
            default_train_config = {
                "lr": 1e-3,
                "num_epochs": 50,
                "batch_size": 64,
                "patience": 5,
                "seed": self.seed,
            }
        else:
            default_train_config = {
                "lr": 1e-3,
                "num_epochs": 30,
                "batch_size": 32,
                "patience": 10,
                "seed": self.seed,
            }

        default_params_dict = {**default_train_config, **self.h_parameters}
        default_model_path = os.path.join(self.results_path, self.model_fpath)
        
        # Perform the training using the configuration
        logging.info(f"Training {self.model_id} with default hyperparameters.")
        self.train_func(config=default_params_dict,
                        data_dir=self.data_config["data_dir"],
                        train_idx=self.train_idx, 
                        valid_idx=self.valid_idx,
                        model_id=self.model_id,
                        model_path=default_model_path,
                        opt_metric=self.opt_metric,
                        device=self.device,
                    )

        default_params_dict['ckpt_path'] = default_model_path
        default_params_dict['model'] = self.model_id

        # Load the best model from the path where it was saved
        model = self._load_model(self.h_parameters, default_model_path)
        
        return model, default_params_dict

    def _load_model(self, hparams_dict, model_path):
        """
        Load the best model from checkpoint.
        
        Parameters:
            hparams_dict (dict): Hyperparameters used in the model.
            model_path (str): Path where the model checkpoint is saved.

        Returns:
            nn.Module: The loaded model.
        """
        try:
            if self.model_id in ['TabMap', 'DeepInsight' ,'IGTD', '1DCNN']:
                best_model = self.model(input_dim=self.data_config["input_size"], 
                                        output_dim=self.data_config["n_classes"], 
                                        **hparams_dict
                                        ).to(self.device)
                best_model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pth')))
            elif self.model_id == 'TabNet':
                best_model = self.model(n_d=hparams_dict["n_a"],
                                        n_a=hparams_dict["n_a"],
                                        n_steps=hparams_dict["n_steps"],
                                        gamma=hparams_dict["gamma"],
                                        lambda_sparse=hparams_dict['lambda_sparse'],
                                        )
                best_model.load_model(os.path.join(model_path, 'tabnet_model.zip'))
            elif self.model_id in ['TabTransformer']:
                best_model = self.model(categories=(),
                                        num_continuous=self.data_config["input_size"], 
                                        dim_out=self.data_config["n_classes"], 
                                        mlp_hidden_mults=(4, 2),
                                        dim=32,
                                        mlp_act=nn.ReLU(),
                                        depth=hparams_dict["depth"],
                                        heads=hparams_dict["heads"],
                                        attn_dropout=hparams_dict["attn_dropout"],
                                        ff_dropout=hparams_dict["ff_dropout"],
                                        ).to(self.device)
                best_model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pth')))
            return best_model
        except Exception as e:
            logging.error("Failed to load model: %s", e)
            return None
    
    def _best_model(self):
        """
        Retrieve the best model configuration and load it.
        
        Returns:
            tuple: (best_model, best_params_dict)
        """
        best_trial = self.cv_results.best_trial
        logging.info(f'Best trial result: {best_trial.value}')

        best_params_dict = best_trial.params
        best_h_params_dict = {k: v for k, v in best_params_dict.items() if k in self.h_parameters}
        best_params_dict['ckpt_path'] = os.path.join(self.results_path, self.model_fpath, f'trial_{best_trial._trial_id}')
        best_params_dict['model'] = self.model_id
        best_model = self._load_model(best_h_params_dict, best_params_dict['ckpt_path'])
        return best_model, best_params_dict

    def _remove_ckpts(self):
        """
        Remove all checkpoint directories except for the best model's.
        """
        save_dir = os.path.join(self.results_path, self.model_fpath)
        best_trial = self.cv_results.best_trial
        best_trial_path = f'trial_{best_trial._trial_id}'
        try:
            for ckpt_dir in os.listdir(save_dir):
                ckpt_path = os.path.join(save_dir, ckpt_dir)
                if os.path.isdir(ckpt_path) and ckpt_dir != best_trial_path:
                    shutil.rmtree(ckpt_path)
                    logging.info(f"Removed checkpoint directory: {ckpt_dir}")
        except OSError as e:
            logging.error("Failed to remove checkpoint directories: %s", e)