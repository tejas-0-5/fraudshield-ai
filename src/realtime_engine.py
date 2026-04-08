"""
Real-Time Simulation Engine
============================
Simulates a streaming transaction pipeline:
- Processes transactions one-by-one (or in micro-batches)
- Applies configurable fraud threshold
- Generates risk alerts: SAFE / SUSPICIOUS / FRAUD
- Logs all predictions for monitoring
"""

import numpy as np
import pandas as pd
import time
import logging
import os
import json
from datetime import datetime
from typing import Iterator, Optional
from src.explainability import prob_to_score, score_to_risk, score_to_alert_level

logger = logging.getLogger(__name__)

LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)


class RealTimeSimulator:
    """
    Simulates a real-time fraud detection pipeline.
    Supports configurable threshold, streaming speed, and alert callbacks.
    """
    
    def __init__(self, model, feature_columns: list, threshold: float = 0.5):
        """
        Args:
            model: Trained sklearn model with predict_proba
            feature_columns: List of feature column names
            threshold: Decision boundary (0.0–1.0)
        """
        self.model = model
        self.feature_columns = feature_columns
        self.threshold = threshold
        self.prediction_log = []
        self.alert_counts = {'SAFE': 0, 'SUSPICIOUS': 0, 'FRAUD': 0}
        self.running_stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'avg_score': 0.0,
            'high_risk_streaks': 0
        }
    
    def set_threshold(self, threshold: float):
        """Adjust decision threshold at runtime (business logic tuning)."""
        old = self.threshold
        self.threshold = max(0.01, min(0.99, threshold))
        logger.info(f"Threshold updated: {old:.2f} → {self.threshold:.2f}")
    
    def predict_transaction(self, features: np.ndarray, tx_id: str = None) -> dict:
        """
        Process a single transaction and return full alert payload.
        
        Args:
            features: Feature vector (1D array matching feature_columns)
            tx_id: Optional transaction identifier
        
        Returns:
            Alert dict with risk level, score, and metadata
        """
        x = features.reshape(1, -1)
        
        # Get fraud probability
        if hasattr(self.model, 'predict_proba'):
            prob = float(self.model.predict_proba(x)[0][1])
        else:
            prob = 0.5
        
        score = prob_to_score(prob)
        alert = score_to_alert_level(score)
        
        # Apply threshold for decision
        decision = 'BLOCK' if prob >= self.threshold else alert['action']
        
        result = {
            'tx_id': tx_id or f"TX_{int(time.time() * 1000)}",
            'timestamp': datetime.now().isoformat(),
            'fraud_probability': prob,
            'fraud_score': score,
            'risk_level': alert['risk_level'],
            'decision': decision,
            'threshold_used': self.threshold,
            'alert': alert
        }
        
        # Update running statistics
        self._update_stats(result)
        self._log_prediction(result)
        
        return result
    
    def stream_transactions(self, df: pd.DataFrame,
                            delay_ms: float = 0,
                            max_transactions: int = None) -> Iterator[dict]:
        """
        Generator: yields predictions one-by-one, simulating a stream.
        
        Args:
            df: DataFrame with feature columns
            delay_ms: Simulated processing delay in milliseconds
            max_transactions: Limit number of transactions (None = all)
        
        Yields:
            Alert dict per transaction
        """
        n = min(len(df), max_transactions or len(df))
        feature_data = df[self.feature_columns].values[:n]
        
        logger.info(f"Starting stream simulation: {n} transactions")
        
        for i in range(n):
            result = self.predict_transaction(feature_data[i], tx_id=f"TX_{i+1:06d}")
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
            yield result
    
    def _update_stats(self, result: dict):
        """Update running statistics after each prediction."""
        self.running_stats['total_processed'] += 1
        if result['risk_level'] == 'FRAUD':
            self.running_stats['fraud_detected'] += 1
        
        n = self.running_stats['total_processed']
        prev_avg = self.running_stats['avg_score']
        self.running_stats['avg_score'] = (prev_avg * (n - 1) + result['fraud_score']) / n
        
        self.alert_counts[result['risk_level']] = self.alert_counts.get(result['risk_level'], 0) + 1
    
    def _log_prediction(self, result: dict):
        """Append prediction to in-memory log (also write to file)."""
        log_entry = {k: v for k, v in result.items() if k != 'alert'}
        self.prediction_log.append(log_entry)
        
        # Write to daily log file
        log_file = os.path.join(LOGS_DIR, f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_dashboard_stats(self) -> dict:
        """Return real-time dashboard statistics."""
        return {
            **self.running_stats,
            'alert_distribution': self.alert_counts,
            'fraud_rate_pct': round(
                self.running_stats['fraud_detected'] /
                max(self.running_stats['total_processed'], 1) * 100, 2
            ),
            'current_threshold': self.threshold,
            'recent_alerts': self.prediction_log[-20:]  # Last 20
        }
    
    def get_prediction_log_df(self) -> pd.DataFrame:
        """Return prediction log as DataFrame for analysis."""
        if not self.prediction_log:
            return pd.DataFrame()
        return pd.DataFrame(self.prediction_log)


def generate_synthetic_fraud_scenarios(n_legitimate: int = 50,
                                       n_fraud: int = 10,
                                       feature_dim: int = 38,
                                       random_state: int = 42) -> pd.DataFrame:
    """
    Synthetic Fraud Scenario Generator.
    
    Creates realistic-looking test transactions by sampling from
    known distribution patterns of the credit card dataset.
    
    Fraud patterns simulated:
    1. High-amount transactions (card-present fraud)
    2. Unusual time patterns (late-night transactions)
    3. Anomalous V-feature values (PCA outliers)
    4. Rapid successive transactions (velocity fraud)
    """
    rng = np.random.RandomState(random_state)
    scenarios = []
    
    # Legitimate transactions
    for i in range(n_legitimate):
        tx = {
            'type': 'LEGITIMATE',
            'amount': rng.lognormal(3, 1.5),  # ~$20-200
            'time_hours': rng.uniform(8, 22),  # Business hours
            **{f'V{j}': rng.normal(0, 1) for j in range(1, 29)},
            'scenario': 'Normal purchase'
        }
        scenarios.append(tx)
    
    # Fraud Scenario 1: High-value card fraud
    for i in range(n_fraud // 4):
        tx = {
            'type': 'FRAUD',
            'amount': rng.uniform(1000, 10000),
            'time_hours': rng.uniform(0, 6),    # Late night
            'V14': rng.uniform(-15, -8),          # Strong fraud signal
            'V17': rng.uniform(-10, -5),
            **{f'V{j}': rng.normal(-2, 2) for j in [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,18,19,20,21,22,23,24,25,26,27,28]},
            'scenario': 'High-value card fraud (late night)'
        }
        scenarios.append(tx)
    
    # Fraud Scenario 2: Small test transactions (card testing)
    for i in range(n_fraud // 4):
        tx = {
            'type': 'FRAUD',
            'amount': rng.uniform(0.01, 5),
            'time_hours': rng.uniform(2, 5),
            'V14': rng.uniform(-12, -6),
            'V4': rng.uniform(4, 8),
            **{f'V{j}': rng.normal(-1, 3) for j in [1,2,3,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28]},
            'scenario': 'Card testing (micro-transaction)'
        }
        scenarios.append(tx)
    
    # Fraud Scenario 3: PCA outlier fraud
    for i in range(n_fraud // 4):
        tx = {
            'type': 'FRAUD',
            'amount': rng.uniform(100, 500),
            'time_hours': rng.uniform(0, 24),
            **{f'V{j}': rng.normal(-5, 4) for j in range(1, 29)},
            'scenario': 'Anomalous behavioral pattern'
        }
        scenarios.append(tx)
    
    # Fraud Scenario 4: Velocity fraud (duplicate-like)
    for i in range(n_fraud - 3 * (n_fraud // 4)):
        tx = {
            'type': 'FRAUD',
            'amount': rng.uniform(50, 300),
            'time_hours': rng.uniform(12, 18),
            'V17': rng.uniform(-12, -4),
            'V14': rng.uniform(-8, -3),
            **{f'V{j}': rng.normal(0, 2) for j in [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,18,19,20,21,22,23,24,25,26,27,28]},
            'scenario': 'Repeated transaction fraud'
        }
        scenarios.append(tx)
    
    df = pd.DataFrame(scenarios)
    # Ensure all V columns exist
    for j in range(1, 29):
        col = f'V{j}'
        if col not in df.columns:
            df[col] = 0.0
    
    return df.reset_index(drop=True)
