import argparse
import os

import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional
# from pyDeepInsight import ImageTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import DL_MODELS, DL_MODELS_IMAGE_BASED, ML_MODELS, METRICS
from dataloader.dataset import load_data
from hparams_tuner.ml_models_tuner import MLModelTuner
from hparams_tuner.dl_models_tuner import DLModelTuner
from evaluate_model import Model_Evaluation
from tabmap_construction import TabMapGenerator

# Construct argument parser
def get_args():
    parser = argparse.ArgumentParser(description='Script for training a TabMap classifier',
                                     add_help=False)

    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='/home/yan/TabMap/data', 
                        help='Path to the dataset directory')
    parser.add_argument('--data_set', type=str, default='micromass',
                        help='Dataset name')
    parser.add_argument('--scaler_name', type=str, default='minmax', 
                        help='Data preprocessing scaler name')
    
    # TabMap construction parameters
    parser.add_argument('--metric', type=str, default='euclidean', choices=['correlation', 'euclidean', 'gower'],
                        help='Metric used for calculating feature associations')
    parser.add_argument('--loss_fun', type=str, default='kl_loss', choices=['kl_loss', 'square_loss', 'sqeuclidean'],
                        help='Loss function used during TabMap construction')
    parser.add_argument('--epsilon', type=float, default=0.0, choices=[0.0, 0.1, 1.0, 10.0],
                        help='Regularization term for TabMap construction. \
                            Set to 0 for no regularization. \
                            Use a positive value for regularization strength.')
    parser.add_argument('--num_iter', type=int, default=10,
                        help='Number of iterations to learn the coupling matrix')
    parser.add_argument('--tabmap_version', type=str, default='v2.0',
                        help='TabMap version')
    
    # Hyperparameter tuning paramters
    parser.add_argument('--use_default_hparams', action="store_true", default=True,
                        help='Use default hyperparameters instead of tuning')
    parser.add_argument('--opt_metric', type=str, default='balanced_accuracy', choices=['balanced_accuracy', 'loss'],
                        help='Metric to optimize during hyperparameter tuning')
    
    # Evaluation parameters
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of cross-validation trials')
    
    # Other parameters
    parser.add_argument('--results_path', type=str, default='/home/yan/TabMap/results', 
                        help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for training and inference')
    parser.add_argument("--seed", default=0, type=int)
    
    return parser.parse_args()


def generate_images(args, 
                    model_id: str, 
                    features: np.ndarray, 
                    train_idx: list, 
                    test_idx: list,
                    save_path: Optional[str]=None, 
                    save_images: bool=True) -> Tuple[np.ndarray, float]:
    print(f"Generating images for {model_id}")

    if model_id not in DL_MODELS_IMAGE_BASED:
        raise ValueError(f"Unsupported model_id: {model_id}")
    
    if model_id == 'TabMap':
        generator = TabMapGenerator(metric=args.metric, 
                                    loss_fun=args.loss_fun,
                                    epsilon=args.epsilon, 
                                    version=args.tabmap_version,
                                    num_iter=args.num_iter)
        generator.fit(features[train_idx], truncate=False)
        X_train_img = generator.transform(features[train_idx])
        X_test_img = generator.transform(features[test_idx])
    
    elif model_id == 'DeepInsight':
        generator = ImageTransformer(feature_extractor='tsne', pixels=(50, 50))
        generator.fit(features[train_idx])
        X_train_img = generator.transform(features[train_idx], img_format='scalar')
        X_test_img = generator.transform(features[test_idx], img_format='scalar')
    
    images = np.empty((len(features), X_train_img.shape[1], X_train_img.shape[2]))
    images[train_idx] = X_train_img
    images[test_idx] = X_test_img

    if save_images and save_path:
        # this is recommended as our model training will load saved files.
        np.save(save_path, images) 
        print(f"Images saved to {save_path}")
    
    return images


def main(args):
    model_list = ['TabMap', 'LR', 'RF', 'GB', 'XGB']
    # model_list = ['TabMap', 'TabTransformer', '1DCNN', 'LR', 'RF', 'GB', 'XGB']
    
    # Define paths and ensure directory existence
    data_dir = os.path.join(args.data_path, args.data_set)
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)
    
    # Load the dataset
    features, labels, _ = load_data(data_dir, scaler_name=args.scaler_name, preprocessed=False)
    n_classes = labels.shape[1] if labels.ndim > 1 else len(np.unique(labels))

    # Initialize dataframes for storing results
    predictions_test_df = pd.DataFrame()
    performance_test_df = pd.DataFrame()
    
    # Run trials
    for trial in range(args.num_trials):        
        args.seed = trial
        
        # set seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        for fold_id, (train_idx_all, test_idx) in enumerate(skf.split(features, labels)):
            print(f'\nTrail: {trial}, Fold_id: {fold_id}')
            train_idx, valid_idx = train_test_split(train_idx_all, test_size=0.125,
                                                    random_state=args.seed,
                                                    stratify=labels[train_idx_all])

            images_dict = {}
            for model_id in model_list:
                print(f"\nTraining {model_id}...")
                if model_id in DL_MODELS_IMAGE_BASED:
                    images_save_path = os.path.join(data_dir, f"{model_id}.npy")
                    images = generate_images(args, model_id, features, train_idx_all, test_idx, 
                                             save_path=images_save_path)
                    images_dict[model_id] = images

                data_config = {
                    "data_set": args.data_set,
                    "data_dir": data_dir,
                    "n_classes": n_classes,
                    "input_size": images_dict[model_id].shape[1:] \
                        if model_id in DL_MODELS_IMAGE_BASED else features.shape[1],
                    "fold_id": fold_id,
                }
                
                if model_id in DL_MODELS:
                    tuner = DLModelTuner(data_config, train_idx, valid_idx, model_id, 
                                        args.results_path, 
                                        use_default_hparams=args.use_default_hparams,
                                        opt_metric=args.opt_metric,
                                        random_seed=args.seed)
                elif model_id in ML_MODELS:
                    tuner = MLModelTuner(data_config, train_idx, valid_idx, model_id, 
                                        args.results_path, 
                                        use_default_hparams=args.use_default_hparams,
                                        random_seed=args.seed)
                
                best_params_dict = tuner.params_dict
                best_params_dict['trial'] = trial
                best_params_dict['fold'] = fold_id
                
                # Model evaluation on the best trained model
                model_eval = Model_Evaluation(model_id)
                if model_id in DL_MODELS_IMAGE_BASED:
                    _, y_pred = model_eval.model_predict(tuner.final_model, 
                                                         features=images_dict[model_id][test_idx])
                else:
                    _, y_pred = model_eval.model_predict(tuner.final_model,
                                                         features=features[test_idx])
                
                predictions_test = pd.DataFrame({"model": model_id, "fold": fold_id, "trial": trial,  
                                                 "ground_truth": labels[test_idx].ravel(), "pred": y_pred.ravel()})
                predictions_test_df = pd.concat([predictions_test_df, predictions_test], ignore_index=True)
                
                performance_test = model_eval.prediction_performance(labels[test_idx].ravel(), y_pred.ravel())
                performance_test = pd.DataFrame([performance_test])
                performance_test["trial"] = trial
                performance_test["fold"] = fold_id
                performance_test_df = pd.concat([performance_test_df, performance_test])

                # Save the performance
                print('Saving the prediction results...')
                predictions_test_df.set_index(["model", "trial", "fold"]).to_csv(f"{args.results_path}/{args.data_set}/model_preds.csv", sep='\t')
                performance_test_df.set_index(["model", "trial", "fold"]).to_csv(f"{args.results_path}/{args.data_set}/model_performance.csv", sep='\t')
            
            print(f'\n', performance_test_df)
    
    predictions_test_df.set_index(["model", "trial", "fold"]).to_csv(f"{args.results_path}/{args.data_set}/model_preds.csv", sep='\t')
    performance_test_df.set_index(["model", "trial", "fold"]).to_csv(f"{args.results_path}/{args.data_set}/model_performance.csv", sep='\t')
    mean_performance_test_df = performance_test_df.groupby(["model"])[METRICS].agg(["mean", "std"]).reset_index()
    
    for metric in METRICS:
        mean_performance_test_df[(metric, "mean")] = mean_performance_test_df[(metric, "mean")].round(4)
        mean_performance_test_df[(metric, "std")] = mean_performance_test_df[(metric, "std")].round(4)
    mean_performance_test_df.to_csv(f"{args.results_path}/{args.data_set}/mean_model_performance.csv", sep='\t')
    print("\nMean Performance Metrics:")
    print(mean_performance_test_df.to_string(index=False), '\n')  


if __name__ == '__main__':
    args = get_args()
    main(args)
