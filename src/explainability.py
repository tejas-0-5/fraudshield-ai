"""
Explainable AI Module
=====================
Provides global and local model explanations without external SHAP dependency.
Uses:
- Global: Tree-based feature importances (Random Forest / GB)
- Local: Perturbation-based LIME-style explanation
- Risk scoring: 0-100 fraud probability scale
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_global_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract global feature importance from tree-based models.
    Returns DataFrame sorted by importance descending.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Logistic Regression: use absolute coefficient values
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model doesn't support feature importance extraction.")
    
    # Normalize to sum to 1
    importances = importances / importances.sum()
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    df['importance_pct'] = (df['importance'] * 100).round(2)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def explain_transaction_local(model, X_instance: np.ndarray,
                               feature_names: list,
                               n_perturbations: int = 500,
                               random_state: int = 42) -> dict:
    """
    Local perturbation-based explanation (LIME-style).
    
    How it works:
    1. Perturb the input sample by adding Gaussian noise
    2. Get predictions for all perturbed samples
    3. Measure how each feature's variation correlates with prediction change
    4. Rank features by their local impact
    
    Args:
        model: Trained classifier with predict_proba
        X_instance: Single sample (1D array)
        feature_names: List of feature names
        n_perturbations: Number of perturbed samples
        random_state: Reproducibility
    
    Returns:
        dict with feature contributions and explanation text
    """
    rng = np.random.RandomState(random_state)
    x = X_instance.reshape(1, -1)
    
    # Get baseline prediction
    if hasattr(model, 'predict_proba'):
        base_prob = model.predict_proba(x)[0][1]
    else:
        base_prob = float(x[0][0])  # fallback
    
    # Generate perturbed samples
    noise_scale = np.std(X_instance) * 0.1 + 1e-8
    perturbations = rng.normal(0, noise_scale, size=(n_perturbations, len(X_instance)))
    X_perturbed = x + perturbations
    
    # Get predictions for all perturbed samples
    if hasattr(model, 'predict_proba'):
        perturbed_probs = model.predict_proba(X_perturbed)[:, 1]
    else:
        perturbed_probs = np.full(n_perturbations, base_prob)
    
    # Compute correlation of each feature's perturbation with prediction change
    pred_changes = perturbed_probs - base_prob
    contributions = []
    for i in range(len(feature_names)):
        feat_perturbations = perturbations[:, i]
        if feat_perturbations.std() > 1e-10:
            corr = np.corrcoef(feat_perturbations, pred_changes)[0, 1]
        else:
            corr = 0.0
        contributions.append(float(corr if not np.isnan(corr) else 0.0))
    
    # Scale contributions to sum to base_prob (approximate)
    contributions = np.array(contributions)
    
    result_df = pd.DataFrame({
        'feature': feature_names,
        'contribution': contributions,
        'value': X_instance,
        'abs_contribution': np.abs(contributions)
    }).sort_values('abs_contribution', ascending=False)
    
    return {
        'fraud_probability': float(base_prob),
        'fraud_score': prob_to_score(base_prob),
        'risk_level': score_to_risk(prob_to_score(base_prob)),
        'top_features': result_df.head(10).to_dict('records'),
        'explanation_text': _generate_explanation_text(result_df.head(5), base_prob)
    }


def _generate_explanation_text(top_features: pd.DataFrame, fraud_prob: float) -> str:
    """Generate human-readable explanation for why a transaction is flagged."""
    lines = []
    risk = score_to_risk(prob_to_score(fraud_prob))
    lines.append(f"Risk Assessment: {risk} (Fraud Probability: {fraud_prob:.1%})")
    lines.append("")
    lines.append("Key Factors Driving This Prediction:")
    
    for _, row in top_features.iterrows():
        direction = "↑ increases" if row['contribution'] > 0 else "↓ decreases"
        strength = "strongly" if abs(row['contribution']) > 0.3 else "moderately" if abs(row['contribution']) > 0.1 else "slightly"
        lines.append(f"  • {row['feature']}: value={row['value']:.3f} — {strength} {direction} fraud risk")
    
    return "\n".join(lines)


def prob_to_score(prob: float) -> float:
    """Convert fraud probability [0,1] to fraud score [0,100]."""
    return round(min(max(prob * 100, 0), 100), 1)


def score_to_risk(score: float) -> str:
    """
    Risk categorization:
    0–30   → Safe (Low Risk)
    30–70  → Suspicious (Medium Risk)
    70–100 → Fraud (High Risk)
    """
    if score < 30:
        return "SAFE"
    elif score < 70:
        return "SUSPICIOUS"
    else:
        return "FRAUD"


def score_to_alert_level(score: float) -> dict:
    """Return full alert metadata for a fraud score."""
    risk = score_to_risk(score)
    colors = {'SAFE': '#22c55e', 'SUSPICIOUS': '#f59e0b', 'FRAUD': '#ef4444'}
    icons = {'SAFE': '✅', 'SUSPICIOUS': '⚠️', 'FRAUD': '🚨'}
    descriptions = {
        'SAFE': 'Transaction appears legitimate. No action required.',
        'SUSPICIOUS': 'Transaction shows unusual patterns. Manual review recommended.',
        'FRAUD': 'HIGH PROBABILITY of fraudulent activity. Block and investigate immediately.'
    }
    return {
        'risk_level': risk,
        'score': score,
        'color': colors[risk],
        'icon': icons[risk],
        'description': descriptions[risk],
        'action': {
            'SAFE': 'APPROVE',
            'SUSPICIOUS': 'REVIEW',
            'FRAUD': 'BLOCK'
        }[risk]
    }


def batch_explain(model, X_batch: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Generate fraud scores and risk levels for a batch of transactions."""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_batch)[:, 1]
    else:
        probs = np.zeros(len(X_batch))
    
    scores = [prob_to_score(p) for p in probs]
    risks = [score_to_risk(s) for s in scores]
    
    return pd.DataFrame({
        'fraud_probability': probs,
        'fraud_score': scores,
        'risk_level': risks
    })
