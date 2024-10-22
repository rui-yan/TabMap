import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
# import sys
# sys.path.append('..')
from ..utils.data_util import data_preprocess, add_noise_to_dataset


def load_data(data_dir: str, scaler_name='minmax', preprocessed: bool=True,
              feature_cols=None, label_cols=None, noise_level=0, verbose=False):
    """
    Load and optionally preprocess data from a specified directory.
    
    Parameters:
    - data_dir (str): The path to the directory containing the data files.
    - scaler_name (str): Scaler to use for feature normalization during preprocessing. Defaults to 'zscore'.
    - preprocessed (bool): Whether the data is already preprocessed. If False, preprocessing will be performed.
    - feature_cols (list, optional): Specific columns to use as features. If None, all columns except labels are used.
    - label_cols (list or str, optional): Specific column(s) to use as labels. If None, the last column is used.
    - noise_level (float): The standard deviation of the noise to add to the dataset. Defaults to 0 (no noise).
    - verbose (bool): If True, the function will print out the label mappings for categorical variables.

    Returns:
    - tuple: Features (numpy array), labels (numpy array), and feature column names (list).
    """
    features_path = os.path.join(data_dir, "features.csv")
    labels_path = os.path.join(data_dir, "labels.csv")
    
    if preprocessed:
        df_features = pd.read_csv(features_path, index_col=0)
        df_labels = pd.read_csv(labels_path, index_col=0)
    else:
        data_set = os.path.basename(data_dir)
        if data_set == 'wdbc':
            data = load_breast_cancer()
            df_features = pd.DataFrame(data['data'], columns=data['feature_names'])
            df_labels = pd.DataFrame(data['target'], columns=['label'])
        else:
            df = pd.read_csv(os.path.join(data_dir, "data_raw.csv"), index_col=0)
            df = df.drop(columns=['id'], errors='ignore')
            
            df_features = df[feature_cols] if feature_cols else df.iloc[:, :-1]
            df_labels = df[label_cols] if label_cols else df.iloc[:, -1]
        
        # Preprocess data if specified
        if scaler_name is not None:
            print("Performing data preprocessing...")
            df_features, df_labels = data_preprocess(df_features, df_labels, scaler_name, verbose=verbose)

        # Optionally add noise to the dataset
        if noise_level > 0:
            print(f"Adding noise level {noise_level}...")
            noisy_features = add_noise_to_dataset(df_features.values, noise_level=noise_level)
            df_features = pd.DataFrame(noisy_features, index=df_features.index, columns=df_features.columns)
        
        # Save the preprocessed data for future use
        df_features.to_csv(os.path.join(data_dir, "features.csv"))
        df_labels.to_csv(os.path.join(data_dir, "labels.csv"))
    
    return df_features.values, df_labels.values.ravel(), df_features.columns.tolist()