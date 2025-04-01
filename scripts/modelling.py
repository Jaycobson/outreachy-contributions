"""
Advanced BBB Permeability Prediction using Multiple Models

This script trains a model to predict blood-brain barrier (BBB) permeability
using featurized datasets. It extracts target values from original data files
and includes data loading, preprocessing, model training, evaluation, and 
visualization with optimized performance tuning.

Command-line usage:
    python script.py --model [xgboost|lightgbm|logistic]
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


# Configure logging to track the training process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('models/bbb_training.log'),
        logging.StreamHandler()
    ]
)

def parse_arguments():
    """Parse command line arguments for model selection."""
    parser = argparse.ArgumentParser(description='Train a BBB permeability prediction model')
    parser.add_argument('--model', type=str, default='xgboost', 
                        choices=['xgboost', 'lightgbm', 'logistic'],
                        help='Model type to use for training (xgboost, lightgbm, or logistic)')
    return parser.parse_args()

def create_directories(model_type):
    """Ensure required directories exist for saving models and results."""
    directories = [
        f'models/{model_type}',
        'data/results',
        'data/plots'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def load_bbb_data():
    """Load featurized data and target values from original data files."""
    datasets = {}

    for split in ['train', 'valid', 'test']:
        # Load featurized data (without target)
        feature_file = f'data/featurized_data/BBB_Martins_{split}_features.csv'
        X = pd.read_csv(feature_file)
        
        # Load original data (with target)
        original_file = f'data/download_data/BBB_Martins_{split}.csv'
        original_df = pd.read_csv(original_file)
        
        # Extract target variable
        y = original_df['Y'].astype(int)  # Assuming 'Y' is the target column
        
        # Verify that the number of samples match
        if len(X) != len(y):
            logging.error(f"Mismatch in number of samples for {split} set: {len(X)} features vs {len(y)} targets")
            raise ValueError(f"Number of samples in featurized and original {split} data don't match")
        
        datasets[split] = {'features': X, 'labels': y}
        logging.info(f"Loaded {split} dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    
    return datasets


def preprocess_data(X_train, X_valid, X_test):
    """Standardize feature data using mean and variance scaling."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.iloc[:,2:])
    X_valid_scaled = scaler.transform(X_valid.iloc[:,2:])
    X_test_scaled = scaler.transform(X_test.iloc[:,2:])
    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler


def create_model(model_type):
    """Create and return a model based on the specified type."""
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=500,
            early_stopping_rounds=50,
            random_state=42,
            use_label_encoder=False
        )
    elif model_type == 'lightgbm':
        model = LGBMClassifier(
            objective='binary',
            metric='auc',
            learning_rate=0.01,
            max_depth=6,
            num_leaves=31,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='liblinear',
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def train_model(model, X_train, y_train, X_valid, y_valid, model_type):
    """Train the selected model with appropriate parameters."""
    if model_type in ['xgboost', 'lightgbm']:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=100 if model_type == 'xgboost' else True
        )
    else:  # Logistic regression
        model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X, y, split_name, model_type):
    """Evaluate the trained model on a dataset and generate performance metrics."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'avg_precision': average_precision_score(y, y_pred_proba)
    }
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_type} - {split_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'data/plots/confusion_matrix_{model_type}_{split_name}.png')
    plt.close()
    
    # Save ROC curve plot
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y, y_pred_proba):.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_type} - {split_name}')
    plt.legend()
    plt.savefig(f'data/plots/roc_curve_{model_type}_{split_name}.png')
    plt.close()
    
    return metrics


def main():
    """Complete pipeline to train, evaluate, and save the model."""
    args = parse_arguments()
    model_type = args.model
    logging.info(f"Selected model type: {model_type}")
    
    create_directories(model_type)
    
    try:
        data = load_bbb_data()
        
        # Preprocess features
        X_train, X_valid, X_test, scaler = preprocess_data(
            data['train']['features'], data['valid']['features'], data['test']['features']
        )
        y_train, y_valid, y_test = data['train']['labels'], data['valid']['labels'], data['test']['labels']
        
        # Create and train model
        logging.info(f"Training {model_type} model...")
        model = create_model(model_type)
        model = train_model(model, X_train, y_train, X_valid, y_valid, model_type)
        
        # Save model using pickle
        model_path = f'models/{model_type}/{model_type}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved successfully to {model_path}")
        
        # Also save the scaler for future use
        scaler_path = f'models/{model_type}/feature_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"Feature scaler saved to {scaler_path}")
        
        # Evaluate model and store results
        results = {}
        for split, X, y in [('train', X_train, y_train), ('valid', X_valid, y_valid), ('test', X_test, y_test)]:
            logging.info(f"Evaluating {split} set...")
            results[split] = evaluate_model(model, X, y, split, model_type)
            logging.info(f"{split} set metrics: {results[split]}")
        
        # Save results to JSON
        with open(f'data/results/model_evaluation_{model_type}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()