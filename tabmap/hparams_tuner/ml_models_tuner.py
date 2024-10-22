import logging
import os
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ..dataloader.dataset import load_data
from ..config import ML_MODELS


class MLModelTuner:
    def __init__(self, data_config, train_idx, valid_idx, model_id, fold_id=None, results_path=None, 
                use_default_hparams=False, opt_metric='balanced_accuracy', random_seed=None, cv_fold=5):

        self.data_config = data_config
        self.train_idx, self.valid_idx = train_idx, valid_idx
        self.model_id = model_id
        self.opt_metric = opt_metric
        self.results_path = results_path
        self.cv_fold = cv_fold
        
        if fold_id is not None:
            self.model_fpath = f"{data_config['data_set']}/{data_config['data_set']}_{fold_id}/{model_id}"
        else:
            self.model_fpath = f"{data_config['data_set']}/{data_config['data_set']}/{model_id}"
        
        # Set up environment
        self.seed = random_seed
        self.features, self.labels, _ = load_data(self.data_config["data_dir"])
        self.model = self._get_ml_model()

        if use_default_hparams:
            self.params_dict = {}
            self.final_model = self._fit_model(self.model)
        else:
            self.h_parameters = self._hyperparameters()
            self.cv_results = self._cross_validation()
            self.final_model, self.params_dict = self._best_model()
    
    def _get_ml_model(self):
        """
        Retrieve the deep learning model and training function based on model ID.
        """
        model_registry = {
            'LR': LogisticRegression(), 
            'RF': RandomForestClassifier(), 
            'GB': GradientBoostingClassifier(), 
            'XGB': XGBClassifier()
        }
        
        if self.model_id in ML_MODELS:
            return model_registry.get(self.model_id, None)
        else:
            raise ValueError(f"Invalid model_id: {self.model_id}")
    
    def _hyperparameters(self):
        """
        Define hyperparameters for model training.
        """
        optimized_hparams = {
            "LR": 
                {
                    'penalty': ['l2'], 
                    'C': [0.01, 0.1, 1, 10],
                    'max_iter': [1000]
                },
            "RF": 
                {
                    'n_estimators': [25, 100, 200], 
                    'max_features': ['sqrt'],
                    'min_samples_split': [2, 3, 5], 
                    'min_samples_leaf': [1, 2, 5]
                 },
            "GB": 
                {
                    'n_estimators': [50, 100, 150], 
                    'learning_rate': [0.01, 0.1, 0.2], 
                    'max_depth': [3, 5, 7]
                },
            "XGB": 
                {
                    'n_estimators': [100, 200, 300], 
                    'learning_rate': [0.01, 0.1, 0.2], 
                    'max_depth': [3, 5, 7]}
        }
        
        return optimized_hparams.get(self.model_id, {})
    
    def _cross_validation(self):
        """
        Configure and execute a cross-validation study to optimize model hyperparameters.
        """
        model = self.model if self.labels.ndim == 1 else self.MultiOutputClassifier(self.model)
        cv_results = GridSearchCV(model, param_grid=self.h_parameters, cv=self.cv_fold,
                                  scoring=self.opt_metric, n_jobs=-1)
        all_indices = np.concatenate((self.train_idx, self.valid_idx))
        cv_results.fit(self.features[all_indices], self.labels[all_indices])
        return cv_results
    
    def _fit_model(self, model):
        model.fit(self.features[self.train_idx], self.labels[self.train_idx])
        return model

    def _best_model(self):
        """
        Retrieve the best model configuration and load it.
        """
        best_params_dict = self.cv_results.best_params_
        best_model = self.model.set_params(**best_params_dict, random_state=self.seed)
        best_model = self._fit_model(best_model if self.labels.ndim == 1 else MultiOutputClassifier(best_model))

        model_path = os.path.join(self.results_path, self.model_fpath)
        os.makedirs(model_path, exist_ok=True)
        
        model_filename = os.path.join(model_path, f"{self.model_id}.pkl")
        joblib.dump(best_model, model_filename)
        logging.info(f"Model saved to {model_filename}")
        best_params_dict['model'] = self.model_id
        return best_model, best_params_dict