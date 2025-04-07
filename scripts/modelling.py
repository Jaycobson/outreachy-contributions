import os
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

import shap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from imblearn.over_sampling import SMOTE  # Import SMOTE

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("models/bbb_training.log"),
        logging.StreamHandler()
    ]
)

# ---------------------- Argument Parser ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="BBB Permeability Prediction")
    parser.add_argument("--model", choices=["xgboost", "lightgbm", "logistic"], default="xgboost")
    parser.add_argument("--feat", choices=["eos39co", "eos24ci"], default="eos39co")
    parser.add_argument("--pca", type=float, default=None, help="PCA components (e.g. 100 or 0.95 for 95% variance)")
    parser.add_argument("--shap", action="store_true", help="Enable SHAP analysis")
    parser.add_argument("--smote", action="store_true", help="Enable SMOTE oversampling")
    return parser.parse_args()

# ---------------------- Generate Configuration ID ----------------------
def get_config_id(model_name, featurizer, pca_option, use_smote):
    """Generate a unique configuration ID based on the CLI arguments"""
    config_id = f"{model_name}_{featurizer}"
    if pca_option is not None:
        config_id += f"_pca{pca_option}_"
    if use_smote:
        config_id += "_smote"
    return config_id

# ---------------------- Directory Setup ----------------------
def create_dirs(model_name, featurizer, pca_option, use_smote):
    """Create directories based on configuration ID"""
    config_id = get_config_id(model_name, featurizer, pca_option, use_smote)
    # Create main directories
    os.makedirs(f"models/{model_name}/{config_id}", exist_ok=True)
    os.makedirs(f"data/plots/{model_name}/{config_id}", exist_ok=True)
    os.makedirs(f"data/results/{model_name}/{config_id}", exist_ok=True)
    
    logging.info(f"Created directories for configuration: {config_id}")
    return config_id

# ---------------------- Data Loader ----------------------
def load_data(feat_type):
    data = {}
    for split in ['train', 'valid', 'test']:
        X = pd.read_csv(f"data/featurized_data/featurized_data_with_{feat_type}/BBB_Martins_{split}_features.csv")
        y = pd.read_csv(f"data/download_data/BBB_Martins_{split}.csv")['Y'].astype(int)
        if len(X) != len(y):
            raise ValueError(f"Mismatch in {split}: {len(X)} features vs {len(y)} targets")
        data[split] = {'X': X, 'y': y}
        logging.info(f"{split.title()} loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return data

# ---------------------- Preprocessing with Optional PCA ----------------------
def preprocess(X_train, X_valid, X_test, pca_option=None):
    feature_cols = X_train.columns[2:]  # assume first 2 columns are ID/SMILES

    # Fill missing values with train means
    for col in feature_cols:
        mean = X_train[col].mean()
        X_train[col].fillna(mean, inplace=True)
        X_valid[col].fillna(mean, inplace=True)
        X_test[col].fillna(mean, inplace=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_valid_scaled = scaler.transform(X_valid[feature_cols])
    X_test_scaled = scaler.transform(X_test[feature_cols])

    if pca_option is not None:
        pca = PCA(n_components=pca_option)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_valid_scaled = pca.transform(X_valid_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        logging.info(f"PCA reduced features to {X_train_scaled.shape[1]} dimensions")

    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler

# ---------------------- Model Factory ----------------------
def get_model(name):
    if name == "xgboost":
        return xgb.XGBClassifier(
            learning_rate=0.03, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, n_estimators=400,
            random_state=42, use_label_encoder=False, eval_metric='auc'
        )
    elif name == "lightgbm":
        return LGBMClassifier(
            learning_rate=0.03, max_depth=6, num_leaves=31,
            n_estimators=400, subsample=0.9,
            colsample_bytree=0.9, random_state=42
        )
    elif name == "logistic":
        return LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42)

# ---------------------- Training ----------------------
def train_model(model, X_train, y_train, X_valid, y_valid, name, use_smote=False):
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)  # Apply SMOTE
        logging.info(f"SMOTE applied: {X_train.shape[0]} samples")

    if name in ["xgboost", "lightgbm"]:
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)])
    else:
        model.fit(X_train, y_train)
    return model
# ---------------------- Evaluation ----------------------
def evaluate(model, X, y, split, config_id, model_name):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        # 'precision': precision_score(y, y_pred),
        # 'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        # 'roc_auc': roc_auc_score(y, y_prob),
        # 'avg_precision': average_precision_score(y, y_prob)
    }

    logging.info(f"[{split.upper()}] Metrics: " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{split.upper()} Confusion Matrix - {config_id}")
    plt.savefig(f"data/plots/{model_name}/{config_id}/confusion_matrix_{split}.png")
    plt.close()

    return metrics

# ---------------------- SHAP Analysis ----------------------
def shap_analysis(model, X_train, feature_cols, config_id, model_name):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    
    # Create a figure that won't be displayed
    plt.figure(figsize=(10, 8))
    
    # Generate the SHAP plot with show=False to prevent display
    shap.summary_plot(shap_values, X_train, feature_names=feature_cols, show=False)
    
    # Save the figure
    plt.savefig(f"data/plots/{model_name}/{config_id}/shap_summary_plot.png", dpi=150, bbox_inches='tight')
    
    # Close the figure to prevent display and free memory
    plt.close()

# ---------------------- Save Results to JSON ----------------------
def save_results(results, config_id, model_name):
    """Save results to JSON file"""
    filename = f"data/results/{model_name}/{config_id}/metrics.json"
    
    # Save results dictionary to JSON file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Results saved to {filename}")

# ---------------------- Save Config Details ----------------------
def save_config_details(args, config_id,model_name):
    """Save configuration details to a separate file"""
    config_details = {
        "model_type": args.model,
        "featurizer": args.feat,
        "pca": args.pca,
        "smote": args.smote,
        "config_id": config_id
    }
    
    # Save configuration to JSON
    with open(f"data/results/{model_name}/{config_id}/config.json", 'w') as f:
        json.dump(config_details, f, indent=4)
    
    # Also save as a text file for quick reference
    with open(f"data/results/{model_name}/{config_id}/config.txt", 'w') as f:
        for key, value in config_details.items():
            f.write(f"{key}: {value}\n")

def main():
    args = parse_args()
    
    # Create unique configuration ID and directories
    config_id = create_dirs(args.model, args.feat, args.pca, args.smote)
    
    # Save configuration details
    save_config_details(args, config_id, args.model)
    
    # Load data
    data = load_data(args.feat)

    X_train, X_valid, X_test = data['train']['X'], data['valid']['X'], data['test']['X']
    y_train, y_valid, y_test = data['train']['y'], data['valid']['y'], data['test']['y']

    X_train_prep, X_valid_prep, X_test_prep, scaler = preprocess(X_train, X_valid, X_test, args.pca)

    model = get_model(args.model)
    model = train_model(model, X_train_prep, y_train, X_valid_prep, y_valid, args.model, use_smote=args.smote)

    # Save model
    with open(f"models/{args.model}/{config_id}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scaler if needed (useful for later inference)
    with open(f"models/{args.model}/{config_id}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Evaluate and collect results
    results = {
        "metrics": {}
    }
    
    results["metrics"]["train"] = evaluate(model, X_train_prep, y_train, "train", config_id, args.model)
    results["metrics"]["valid"] = evaluate(model, X_valid_prep, y_valid, "valid", config_id, args.model)
    results["metrics"]["test"] = evaluate(model, X_test_prep, y_test, "test", config_id, args.model)
    
    # Save results to JSON
    save_results(results, config_id, args.model)

    # SHAP Analysis
    if args.shap:
        feature_cols = X_train.columns[2:]
        shap_analysis(model, X_train_prep, feature_cols, config_id, args.model)
        logging.info("SHAP analysis completed.")
if __name__ == "__main__":
    main()