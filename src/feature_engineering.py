"""
Feature Engineering Layer
==========================
Normalizes raw features, creates velocity/rolling features,
and prepares data for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import logging

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    1. Normalize Amount and Time
    2. Add transaction velocity (rolling count)
    3. Add rolling average amount
    4. Add hour-of-day from Time
    5. Add anomaly indicators (z-score flags)
    
    Returns enriched DataFrame ready for modeling.
    """
    df = df.copy()
    
    # --- 1. Normalize Amount (RobustScaler is outlier-resistant) ---
    scaler_amount = RobustScaler()
    df['Amount_scaled'] = scaler_amount.fit_transform(df[['Amount']])
    
    # --- 2. Normalize Time (convert to hours, then scale) ---
    df['Time_hours'] = df['Time'] / 3600.0
    df['Time_scaled'] = (df['Time_hours'] - df['Time_hours'].mean()) / df['Time_hours'].std()
    
    # --- 3. Hour-of-day (cyclical encoding to capture periodicity) ---
    hour = df['Time_hours'] % 24
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # --- 4. Transaction velocity (rolling 100-row window as proxy for time window) ---
    # Approximates "how many transactions occurred recently"
    df = df.sort_values('Time').reset_index(drop=True)
    df['tx_velocity'] = (
        pd.Series(np.ones(len(df)))
        .rolling(window=100, min_periods=1)
        .sum()
        .values
    )
    
    # --- 5. Rolling average amount (100-tx window) ---
    df['rolling_avg_amount'] = (
        df['Amount']
        .rolling(window=100, min_periods=1)
        .mean()
        .values
    )
    
    # --- 6. Amount deviation from rolling mean (anomaly indicator) ---
    df['amount_deviation'] = (df['Amount'] - df['rolling_avg_amount']).abs()
    
    # --- 7. High-amount flag (top 5% threshold) ---
    high_thresh = df['Amount'].quantile(0.95)
    df['is_high_amount'] = (df['Amount'] > high_thresh).astype(np.int8)
    
    # --- 8. V-feature interaction: top correlated pairs with fraud ---
    # V14 and V17 are known strongest fraud signals in this dataset
    df['V14_V17_interaction'] = df['V14'] * df['V17']
    df['V4_V11_interaction'] = df['V4'] * df['V11']
    
    logger.info(f"Feature engineering complete. Columns: {df.shape[1]}")
    return df


def get_feature_columns() -> list:
    """Return the list of features used for model training."""
    v_features = [f'V{i}' for i in range(1, 29)]
    engineered = [
        'Amount_scaled', 'Time_scaled',
        'hour_sin', 'hour_cos',
        'tx_velocity', 'rolling_avg_amount',
        'amount_deviation', 'is_high_amount',
        'V14_V17_interaction', 'V4_V11_interaction'
    ]
    return v_features + engineered


def get_feature_display_names() -> dict:
    """Human-readable names for features (used in SHAP/explanation display)."""
    names = {f'V{i}': f'PCA Component V{i}' for i in range(1, 29)}
    names.update({
        'Amount_scaled': 'Transaction Amount (normalized)',
        'Time_scaled': 'Transaction Time (normalized)',
        'hour_sin': 'Hour of Day (sin)',
        'hour_cos': 'Hour of Day (cos)',
        'tx_velocity': 'Transaction Velocity (recent 100)',
        'rolling_avg_amount': 'Rolling Avg Amount',
        'amount_deviation': 'Amount Deviation from Average',
        'is_high_amount': 'High-Value Transaction Flag',
        'V14_V17_interaction': 'V14 × V17 Interaction',
        'V4_V11_interaction': 'V4 × V11 Interaction',
    })
    return names


def prepare_X_y(df: pd.DataFrame):
    """Split into features and target arrays."""
    feature_cols = get_feature_columns()
    X = df[feature_cols].values.astype(np.float32)
    y = df['Class'].values.astype(np.int8)
    return X, y, feature_cols
