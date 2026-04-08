"""
Modeling Layer
==============
Trains and compares multiple fraud detection models:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost-equivalent via sklearn)

Includes hyperparameter tuning and comprehensive evaluation.
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, average_precision_score, precision_recall_curve,
    roc_curve
)

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """Stratified split preserving fraud ratio in both sets."""
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def train_logistic_regression(X_train, y_train, class_weights=None):
    """
    Logistic Regression baseline.
    Uses class_weight='balanced' if no explicit weights provided.
    """
    logger.info("Training Logistic Regression...")
    cw = class_weights or 'balanced'
    model = LogisticRegression(
        class_weight=cw,
        max_iter=1000,
        solver='lbfgs',
        C=0.01,           # Regularization (fraud datasets benefit from strong reg)
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return model


def train_random_forest(X_train, y_train, class_weights=None, tune=False):
    """
    Random Forest with optional RandomizedSearchCV hyperparameter tuning.
    """
    logger.info("Training Random Forest...")
    cw = class_weights or 'balanced'
    
    if tune:
        base = RandomForestClassifier(class_weight=cw, random_state=42, n_jobs=-1)
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=10, cv=cv,
            scoring='f1', n_jobs=-1, random_state=42, verbose=0
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        logger.info(f"Best RF params: {search.best_params_}")
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight=cw,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    
    logger.info("Random Forest training complete.")
    return model


def train_gradient_boosting(X_train, y_train, tune=False):
    """
    Gradient Boosting Classifier (XGBoost-equivalent using sklearn).
    Uses subsample + learning rate schedule for regularization.
    For large datasets, uses a subset for speed.
    """
    logger.info("Training Gradient Boosting (XGBoost-equivalent)...")
    
    # For large training sets, subsample to keep training time reasonable
    if len(X_train) > 100_000:
        rng = np.random.RandomState(42)
        # Ensure we keep all minority samples
        fraud_idx = np.where(y_train == 1)[0]
        legit_idx = np.where(y_train == 0)[0]
        
        n_legit_keep = min(len(legit_idx), 50_000)
        legit_sample = rng.choice(legit_idx, size=n_legit_keep, replace=False)
        
        keep = np.concatenate([fraud_idx, legit_sample])
        rng.shuffle(keep)
        X_sub, y_sub = X_train[keep], y_train[keep]
        logger.info(f"GB: Using {len(keep)} samples for training (fraud: {y_sub.sum()})")
    else:
        X_sub, y_sub = X_train, y_train
    
    # Scale pos_weight to handle imbalance (like XGBoost's scale_pos_weight)
    n_neg = (y_sub == 0).sum()
    n_pos = (y_sub == 1).sum()
    scale = n_neg / max(n_pos, 1)
    
    if tune:
        base = GradientBoostingClassifier(random_state=42)
        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'min_samples_split': [5, 10]
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=6, cv=cv,
            scoring='average_precision', n_jobs=-1, random_state=42
        )
        search.fit(X_sub, y_sub)
        model = search.best_estimator_
        logger.info(f"Best GB params: {search.best_params_}")
    else:
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=10,
            random_state=42,
            verbose=0
        )
        model.fit(X_sub, y_sub)
    
    logger.info("Gradient Boosting training complete.")
    return model


def evaluate_model(model, X_test, y_test, model_name: str, threshold: float = 0.5) -> dict:
    """
    Comprehensive model evaluation with all key fraud detection metrics.
    
    Returns dict with all metrics and curve data for visualization.
    """
    # Get probability scores
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        # Normalize to [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
    
    y_pred = (y_prob >= threshold).astype(int)
    
    # Core metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)  # PR-AUC
    cm = confusion_matrix(y_test, y_pred)
    
    # Curve data
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_test, y_prob)
    
    metrics = {
        'model_name': model_name,
        'threshold': threshold,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(avg_precision, 4),
        'confusion_matrix': cm.tolist(),
        'tn': int(cm[0][0]),
        'fp': int(cm[0][1]),
        'fn': int(cm[1][0]),
        'tp': int(cm[1][1]),
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'pr_curve': {'precision': prec_curve.tolist(), 'recall': rec_curve.tolist()},
        'y_prob': y_prob,
        'y_test': y_test,
        'evaluated_at': datetime.now().isoformat()
    }
    
    logger.info(
        f"{model_name} | Precision: {precision:.4f} | Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {avg_precision:.4f}"
    )
    return metrics


def save_model(model, model_name: str, version: str = None) -> str:
    """Save model with versioning using joblib."""
    version = version or datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_v{version}.joblib"
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    
    # Save version manifest
    manifest_path = os.path.join(MODELS_DIR, 'model_manifest.json')
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest[model_name] = {'version': version, 'path': path, 'saved_at': datetime.now().isoformat()}
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Model saved: {path}")
    return path


def load_model(model_name: str):
    """Load the latest version of a model."""
    manifest_path = os.path.join(MODELS_DIR, 'model_manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("No models trained yet. Run training first.")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if model_name not in manifest:
        raise KeyError(f"Model '{model_name}' not found in manifest.")
    return joblib.load(manifest[model_name]['path'])
