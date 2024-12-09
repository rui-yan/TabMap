import numpy as np
import torch
from sklearn import metrics

from .dataloader.data_loader import TabularDataset
from .models.CNN_2d import test_2DCNN
from .models.soft_ordering_CNN_1d import test_1DCNN
# from .models.tabnet import test_tabnet
# from .models.tabtransformer import test_TabTransformer

from .config import DL_MODELS, DL_MODELS_IMAGE_BASED, ML_MODELS


class Model_Evaluation:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def model_predict(self, model, X: np.array):
        """
        Predict using the specified model and features.
        """
        if self.model_id in ML_MODELS:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        elif self.model_id in DL_MODELS:
            testset = TabularDataset(X)
            if self.model_id == '1DCNN':
                test_func = test_1DCNN
            elif self.model_id in DL_MODELS_IMAGE_BASED:
                test_func = test_2DCNN
            # elif self.model_id == 'TabNet':
            #     test_func = test_tabnet
            # elif self.model_id == 'TabTransformer':
            #     test_func = test_TabTransformer
            y_prob, y_pred = test_func(model, testset, self.device)
        return y_prob, y_pred
    
    def prediction_performance(self, y: list, y_pred: list):
        """
        Calculate and return performance metrics for the predictions.
        """
        n_classes = len(set(y))
        average = 'binary' if n_classes == 2 else 'weighted'
        results = {
            "model": self.model_id,
            "BA": metrics.balanced_accuracy_score(y, y_pred),
            "F1": metrics.f1_score(y, y_pred, average=average),
            "MCC": metrics.matthews_corrcoef(y, y_pred),
        }
        results = {key: round(value, 4) if isinstance(value, float) else value for key, value in results.items()}

        return results

