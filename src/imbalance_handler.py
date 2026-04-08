"""
Class Imbalance Handler
========================
Implements SMOTE (oversampling) and random undersampling from scratch
using scikit-learn primitives, since imbalanced-learn may not be available.
Compares both strategies.
"""

import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def smote_oversample(X: np.ndarray, y: np.ndarray,
                     k_neighbors: int = 5,
                     random_state: int = 42) -> tuple:
    """
    SMOTE: Synthetic Minority Over-sampling TEchnique.
    
    For each minority sample, finds k nearest neighbors in minority class,
    then generates synthetic samples along the line segments between them.
    
    Args:
        X: Feature matrix
        y: Target vector (binary 0/1)
        k_neighbors: Number of nearest neighbors to use
        random_state: For reproducibility
    
    Returns:
        X_resampled, y_resampled
    """
    rng = np.random.RandomState(random_state)
    
    # Separate minority and majority
    minority_mask = y == 1
    X_min = X[minority_mask]
    X_maj = X[~minority_mask]
    
    n_to_generate = len(X_maj) - len(X_min)
    logger.info(f"SMOTE: Generating {n_to_generate} synthetic minority samples")
    
    if n_to_generate <= 0:
        return X, y
    
    # Fit k-NN on minority class
    k = min(k_neighbors, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', n_jobs=-1)
    nn.fit(X_min)
    distances, indices = nn.kneighbors(X_min)
    
    # Generate synthetic samples
    synthetic = []
    n_per_sample = n_to_generate // len(X_min) + 1
    
    for i in range(len(X_min)):
        # Pick random neighbors (exclude self at index 0)
        neighbor_indices = indices[i][1:]
        for _ in range(n_per_sample):
            nn_idx = rng.choice(neighbor_indices)
            # Interpolate between sample and chosen neighbor
            alpha = rng.random()
            new_sample = X_min[i] + alpha * (X_min[nn_idx] - X_min[i])
            synthetic.append(new_sample)
            if len(synthetic) >= n_to_generate:
                break
        if len(synthetic) >= n_to_generate:
            break
    
    X_synthetic = np.array(synthetic[:n_to_generate])
    y_synthetic = np.ones(len(X_synthetic), dtype=np.int8)
    
    X_resampled = np.vstack([X, X_synthetic])
    y_resampled = np.concatenate([y, y_synthetic])
    
    # Shuffle
    idx = rng.permutation(len(X_resampled))
    logger.info(f"SMOTE complete. New shape: {X_resampled.shape}, Fraud ratio: {y_resampled.mean():.3f}")
    return X_resampled[idx], y_resampled[idx]


def random_undersample(X: np.ndarray, y: np.ndarray,
                       ratio: float = 1.0,
                       random_state: int = 42) -> tuple:
    """
    Random undersampling: reduce majority class to `ratio` times minority.
    
    Args:
        X: Feature matrix
        y: Target vector
        ratio: majority_count = ratio * minority_count (default 1.0 = balanced)
        random_state: For reproducibility
    
    Returns:
        X_resampled, y_resampled
    """
    rng = np.random.RandomState(random_state)
    
    minority_mask = y == 1
    X_min, y_min = X[minority_mask], y[minority_mask]
    X_maj, y_maj = X[~minority_mask], y[~minority_mask]
    
    n_keep = int(len(X_min) * ratio)
    n_keep = min(n_keep, len(X_maj))
    
    keep_idx = rng.choice(len(X_maj), size=n_keep, replace=False)
    X_maj_under = X_maj[keep_idx]
    y_maj_under = y_maj[keep_idx]
    
    X_resampled = np.vstack([X_min, X_maj_under])
    y_resampled = np.concatenate([y_min, y_maj_under])
    
    # Shuffle
    idx = rng.permutation(len(X_resampled))
    logger.info(f"Undersampling complete. New shape: {X_resampled.shape}, Fraud ratio: {y_resampled.mean():.3f}")
    return X_resampled[idx], y_resampled[idx]


def class_weight_dict(y: np.ndarray) -> dict:
    """
    Compute class weights for use in sklearn models (alternative to resampling).
    Weight = total / (n_classes * count_per_class)
    """
    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)
    n_classes = len(classes)
    weights = {int(c): n_total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    logger.info(f"Class weights: {weights}")
    return weights
