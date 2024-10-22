import numpy as np
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    A dataset class for tabular data. This class handles datasets composed of features and 
    optional labels for tasks like classification or regression.

    Attributes:
        X (np.ndarray): A numpy array containing the input features. for images, 2D arrays (grayscale) expected per item
        y (np.ndarray, optional): Labels corresponding to the input features.
        transform (Callable, optional): Transformation to apply to each feature.
        target_transform (Callable, optional): Transformation to apply to each label.
    """
    
    def __init__(
            self, 
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super(TabularDataset, self).__init__()
        
        self.X = X 
        self.y = y
        self.transform = transform
        self.target_transform = target_transform
        
        if self.X.ndim == 3:
            # for images, input size is the (width, heigth) of images
            self.input_size = self.X.shape[1:]
            self.X = np.expand_dims(self.X, axis=1)  # Add channel dimension
        else:
            # for vectors, input size is the length of vectors
            self.input_size = self.X.shape[1] 
        
        if self.y is not None:
            self.n_classes = len(set(self.y))
    
    def __getitem__(self, index: int):
        """
        Retrieve a single item from the dataset.

        Parameters:
            index (int): The index of the item.

        Returns:
            tuple: (data, target) where data is the feature tensor and target is the label tensor (if available).
        """
        data = torch.from_numpy(self.X[index]).float()
        target = self.y[index] if self.y is not None else []
        
        if self.transform is not None:
            data = self.transform(data)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return data, target

    def __len__(self):
        return len(self.X)