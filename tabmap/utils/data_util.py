import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer, LabelEncoder


def data_preprocess(df_features, df_labels, scaler_name='zscore', truncate=False, verbose=False):
    """
    Preprocesses the data.

    Parameters:
    - df_features (pd.DataFrame): DataFrame containing the feature data.
    - df_labels (pd.Series or pd.DataFrame): Series or DataFrame containing the labels.
    - scaler_name (str): Name of the scaler to use for feature normalization.
    - truncate (bool): If True, truncates features based on variance.
    - verbose (bool): If True, the function will print out the label mappings for categorical variables.
    
    Returns:
    - tuple: Tuple containing two DataFrames, the processed features and labels.
    """
    # Discard features with more than 50% missing values
    df_features = feature_selection(df_features)
    
    # Optional truncation of features
    if truncate:
        df_features = feature_truncation(df_features)
    
    # Impute missing values using the most frequent value
    imputer = SimpleImputer(strategy="most_frequent")
    df_features = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns)

    # Normalize the data using specified scaler
    scaler = {
        'zscore': StandardScaler(),
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler(), 
        'robust': RobustScaler(),
        'norm': Normalizer(), 
        'quantile': QuantileTransformer(output_distribution = 'normal', random_state = 42),
        'power': PowerTransformer()
        }[scaler_name]
    
    df_features = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)
    
    # Encode labels
    le = LabelEncoder()
    df_labels = pd.DataFrame(le.fit_transform(df_labels), columns=['class'])
    
    # Optional: Print the mapping of labels to encoded values
    if verbose:
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        for label, encoded_label in label_mapping.items():
            print(f"{label} -> {encoded_label}")
    
    return df_features, df_labels

def feature_selection(df):
    """
    1. The columns with more than 50% of missing values are discarded.
    2. The columns containing only one value are discarded.
    
    Parameters:
    - df (pd.DataFrame): DataFrame from which to remove features.

    Returns:
    - pd.DataFrame: DataFrame with selected features retained.
    """
    discarded_features = [col for col in df.columns if df[col].isna().mean() > 0.5 or df[col].nunique() <= 1]
    print('Discarded features:', discarded_features)
    cleaned_df = df.drop(columns=discarded_features)
    return cleaned_df

def feature_truncation(df, method='variance'):
    """
    Truncate the features by removing the features with the largest variance.
    This can also be done based on other criteria.
    """
    if method == 'variance':
        num_features_to_keep = int(np.sqrt(df.shape[1])) ** 2
        variance = df.var()
        lowest_variance_cols = variance.nsmallest(df.shape[1] - num_features_to_keep).index
        cleaned_df = df.drop(columns=lowest_variance_cols)
        return cleaned_df
    else:
        raise ValueError(f'Unsupported truncation method: {method}')

def add_noise_to_dataset(dataset, noise_level):
    """
    Adds Gaussian noise to a dataset to simulate data variability.
    
    Parameters:
    - dataset (np.ndarray or pd.DataFrame): The original data.
    - noise_level (float): Standard deviation of the Gaussian noise to be added (0 for no noise).

    Returns:
    - np.ndarray: Dataset with added Gaussian noise, with noise scaled relative to feature ranges.
    """
    if noise_level == 0:
        return dataset

    # Calculate the range for each feature to scale the noise accordingly
    feature_ranges = dataset.max(axis=0) - dataset.min(axis=0)
    noise = np.random.normal(loc=0, scale=noise_level, size=dataset.shape)
    scaled_noise = noise * feature_ranges

    # Add scaled noise and clip to ensure values stay within the [0, 1] range (min-max normalization)
    noisy_dataset = np.clip(dataset + scaled_noise, 0, 1)
    
    return noisy_dataset
