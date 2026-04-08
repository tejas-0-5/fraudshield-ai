"""
Main Training Pipeline
======================
End-to-end fraud detection model training pipeline.

Usage:
    python train_pipeline.py [--tune] [--strategy smote|undersample|weighted]

Steps:
    1. Data ingestion (chunked loading)
    2. Feature engineering
    3. Train/test split
    4. Imbalance handling
    5. Train all models
    6. Evaluate all models
    7. Save best model + all artifacts
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Project imports
sys.path.insert(0, os.path.dirname(__file__))
from src.data_ingestion import load_chunked, get_dataset_stats
from src.feature_engineering import engineer_features, prepare_X_y, get_feature_columns, get_feature_display_names
from src.imbalance_handler import smote_oversample, random_undersample, class_weight_dict
from src.modeling import (
    train_test_split_stratified,
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    evaluate_model,
    save_model
)
from src.explainability import get_global_feature_importance
from src.visualization import (
    plot_class_distribution,
    plot_amount_distribution,
    plot_time_trends,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_feature_importance,
    plot_model_comparison,
    plot_score_distribution,
    plot_threshold_analysis
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log')
    ]
)
logger = logging.getLogger(__name__)

DATA_PATH = '/mnt/user-data/uploads/creditcard.csv'


def run_pipeline(strategy: str = 'weighted', tune: bool = False):
    """
    Full training pipeline.
    
    Args:
        strategy: 'smote' | 'undersample' | 'weighted' (class weights)
        tune: Whether to run hyperparameter tuning
    """
    start = datetime.now()
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION PIPELINE STARTED")
    logger.info(f"Strategy: {strategy} | Tune: {tune}")
    logger.info("=" * 60)
    
    # ──────────────────────────────────────────────────────────
    # STEP 1: Data Ingestion
    # ──────────────────────────────────────────────────────────
    logger.info("\n[1/7] DATA INGESTION")
    df = load_chunked(DATA_PATH, chunksize=50_000)
    stats = get_dataset_stats(df)
    logger.info(f"Dataset stats: {json.dumps({k: v for k, v in stats.items() if k != 'amount_stats'}, indent=2)}")
    
    # Save stats for dashboard
    with open('app/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # ──────────────────────────────────────────────────────────
    # STEP 2: Exploratory Visualizations
    # ──────────────────────────────────────────────────────────
    logger.info("\n[2/7] GENERATING EDA VISUALIZATIONS")
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_trends(df)
    logger.info("EDA plots saved.")
    
    # ──────────────────────────────────────────────────────────
    # STEP 3: Feature Engineering
    # ──────────────────────────────────────────────────────────
    logger.info("\n[3/7] FEATURE ENGINEERING")
    df_engineered = engineer_features(df)
    X, y, feature_cols = prepare_X_y(df_engineered)
    logger.info(f"Features: {len(feature_cols)} | Samples: {X.shape[0]}")
    
    # ──────────────────────────────────────────────────────────
    # STEP 4: Train/Test Split + Imbalance Handling
    # ──────────────────────────────────────────────────────────
    logger.info("\n[4/7] DATA SPLITTING & IMBALANCE HANDLING")
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train fraud: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    
    class_weights = None
    if strategy == 'smote':
        logger.info("Applying SMOTE oversampling...")
        X_train, y_train = smote_oversample(X_train, y_train, k_neighbors=5)
    elif strategy == 'undersample':
        logger.info("Applying random undersampling...")
        X_train, y_train = random_undersample(X_train, y_train, ratio=10.0)
    elif strategy == 'weighted':
        logger.info("Using class weights (no resampling)...")
        class_weights = class_weight_dict(y_train)
    
    # ──────────────────────────────────────────────────────────
    # STEP 5: Train All Models
    # ──────────────────────────────────────────────────────────
    logger.info("\n[5/7] MODEL TRAINING")
    
    models = {}
    
    # Logistic Regression (baseline)
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train, class_weights)
    save_model(models['Logistic Regression'], 'logistic_regression')
    
    # Random Forest
    models['Random Forest'] = train_random_forest(X_train, y_train, class_weights, tune=tune)
    save_model(models['Random Forest'], 'random_forest')
    
    # Gradient Boosting (primary, XGBoost-equivalent)
    models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train, tune=tune)
    save_model(models['Gradient Boosting'], 'gradient_boosting')
    
    # ──────────────────────────────────────────────────────────
    # STEP 6: Evaluation
    # ──────────────────────────────────────────────────────────
    logger.info("\n[6/7] MODEL EVALUATION")
    
    all_metrics = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        
        # Confusion matrix plot
        plot_confusion_matrix(metrics['confusion_matrix'], name)
        
        # Score distribution
        plot_score_distribution(metrics['y_prob'], metrics['y_test'], name)
        
        # Threshold analysis
        plot_threshold_analysis(metrics['y_prob'], metrics['y_test'], name)
    
    # Comparison plots
    plot_roc_curves(all_metrics)
    plot_pr_curves(all_metrics)
    plot_model_comparison(all_metrics)
    
    # Feature importance for best model (Gradient Boosting)
    best_model = models['Gradient Boosting']
    imp_df = get_global_feature_importance(best_model, feature_cols)
    plot_feature_importance(imp_df, top_n=20, model_name='Gradient Boosting')
    
    # Save metrics (excluding large arrays)
    metrics_to_save = []
    for m in all_metrics:
        m_save = {k: v for k, v in m.items()
                  if k not in ('y_prob', 'y_test', 'roc_curve', 'pr_curve')}
        metrics_to_save.append(m_save)
    
    with open('app/model_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2, default=str)
    
    # Save feature importance
    imp_df.to_csv('app/feature_importance.csv', index=False)
    
    # ──────────────────────────────────────────────────────────
    # STEP 7: Summary Report
    # ──────────────────────────────────────────────────────────
    logger.info("\n[7/7] TRAINING SUMMARY")
    logger.info("=" * 60)
    
    best = max(all_metrics, key=lambda m: m['recall'])
    logger.info(f"\nBest model by Recall: {best['model_name']}")
    logger.info(f"  Precision: {best['precision']:.4f}")
    logger.info(f"  Recall:    {best['recall']:.4f}  ← Highest priority metric")
    logger.info(f"  F1:        {best['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {best['roc_auc']:.4f}")
    logger.info(f"  PR-AUC:    {best['pr_auc']:.4f}")
    
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"\nTotal pipeline time: {elapsed:.1f}s")
    logger.info("=" * 60)
    
    return models, all_metrics, feature_cols


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fraud Detection Training Pipeline')
    parser.add_argument('--strategy', choices=['smote', 'undersample', 'weighted'],
                        default='weighted', help='Imbalance handling strategy')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    args = parser.parse_args()
    
    os.makedirs('app', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    run_pipeline(strategy=args.strategy, tune=args.tune)
