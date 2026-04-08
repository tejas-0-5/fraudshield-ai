"""
REST API Backend
================
FastAPI endpoints for the fraud detection platform.

Endpoints:
    POST /predict         - Predict fraud for a transaction
    POST /predict/batch   - Batch predictions
    GET  /explain/{tx_id} - Explain a prediction
    GET  /metrics         - Model performance metrics
    GET  /health          - Health check
    GET  /stats           - Dataset statistics
    POST /threshold       - Update decision threshold
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(__file__))

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from src.modeling import load_model
from src.feature_engineering import get_feature_columns
from src.explainability import (
    explain_transaction_local,
    prob_to_score,
    score_to_risk,
    score_to_alert_level
)
from src.realtime_engine import RealTimeSimulator

logger = logging.getLogger(__name__)

# ── Pydantic Models ─────────────────────────────────────────────

class TransactionInput(BaseModel):
    """Single transaction for fraud prediction."""
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    tx_id: Optional[str] = None


class BatchTransactionInput(BaseModel):
    """Batch of transactions."""
    transactions: List[TransactionInput]


class ThresholdUpdate(BaseModel):
    """Threshold update request."""
    threshold: float
    
    @validator('threshold')
    def validate_threshold(cls, v):
        if not 0.01 <= v <= 0.99:
            raise ValueError("Threshold must be between 0.01 and 0.99")
        return v


class PredictionResponse(BaseModel):
    """Fraud prediction response."""
    tx_id: str
    fraud_probability: float
    fraud_score: float
    risk_level: str
    decision: str
    alert_color: str
    alert_description: str
    timestamp: str


# ── API Application ─────────────────────────────────────────────

def create_app():
    """Factory function to create the FastAPI app with all routes."""
    
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Fraud Detection API",
        description="Enterprise-Grade AI Fraud Detection & Risk Intelligence Platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Load model and simulator at startup
    simulator = None
    feature_cols = get_feature_columns()
    
    @app.on_event("startup")
    async def startup_event():
        nonlocal simulator
        try:
            model = load_model('gradient_boosting')
            simulator = RealTimeSimulator(model, feature_cols, threshold=0.5)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Train the pipeline first.")
    
    def _tx_to_features(tx: TransactionInput) -> np.ndarray:
        """Convert transaction input to engineered feature vector."""
        # Build base features
        raw = {
            'Time': tx.Time, 'Amount': tx.Amount,
            **{f'V{i}': getattr(tx, f'V{i}') for i in range(1, 29)}
        }
        df = pd.DataFrame([raw])
        
        # Apply feature engineering
        from src.feature_engineering import engineer_features, get_feature_columns
        df_eng = engineer_features(df)
        feat_cols = get_feature_columns()
        
        # Fill any missing engineered features with 0
        for col in feat_cols:
            if col not in df_eng.columns:
                df_eng[col] = 0.0
        
        return df_eng[feat_cols].values[0].astype(np.float32)
    
    # ── Health Check ──────────────────────────────────────────
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": simulator is not None,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    
    # ── Single Prediction ─────────────────────────────────────
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(tx: TransactionInput):
        """
        Predict fraud probability for a single transaction.
        Returns risk level, fraud score, and recommended action.
        """
        if simulator is None:
            raise HTTPException(503, "Model not loaded. Run train_pipeline.py first.")
        
        try:
            features = _tx_to_features(tx)
            result = simulator.predict_transaction(features, tx_id=tx.tx_id)
            alert = result['alert']
            
            return PredictionResponse(
                tx_id=result['tx_id'],
                fraud_probability=result['fraud_probability'],
                fraud_score=result['fraud_score'],
                risk_level=result['risk_level'],
                decision=result['decision'],
                alert_color=alert['color'],
                alert_description=alert['description'],
                timestamp=result['timestamp']
            )
        except Exception as e:
            raise HTTPException(500, f"Prediction failed: {str(e)}")
    
    # ── Batch Prediction ──────────────────────────────────────
    @app.post("/predict/batch")
    async def predict_batch(batch: BatchTransactionInput):
        """Predict fraud for a batch of transactions."""
        if simulator is None:
            raise HTTPException(503, "Model not loaded.")
        
        results = []
        for tx in batch.transactions:
            try:
                features = _tx_to_features(tx)
                result = simulator.predict_transaction(features, tx_id=tx.tx_id)
                results.append({
                    'tx_id': result['tx_id'],
                    'fraud_probability': result['fraud_probability'],
                    'fraud_score': result['fraud_score'],
                    'risk_level': result['risk_level'],
                    'decision': result['decision']
                })
            except Exception as e:
                results.append({'tx_id': tx.tx_id, 'error': str(e)})
        
        return {'predictions': results, 'count': len(results)}
    
    # ── Explain ───────────────────────────────────────────────
    @app.post("/explain")
    async def explain(tx: TransactionInput):
        """
        Explain WHY a transaction is flagged as fraud.
        Returns top contributing features with local impact scores.
        """
        if simulator is None:
            raise HTTPException(503, "Model not loaded.")
        
        try:
            features = _tx_to_features(tx)
            explanation = explain_transaction_local(
                simulator.model, features, feature_cols, n_perturbations=200
            )
            return {
                'tx_id': tx.tx_id or 'TX_EXPLAIN',
                **explanation
            }
        except Exception as e:
            raise HTTPException(500, f"Explanation failed: {str(e)}")
    
    # ── Model Metrics ─────────────────────────────────────────
    @app.get("/metrics")
    async def metrics():
        """Return trained model performance metrics."""
        metrics_path = os.path.join(os.path.dirname(__file__), 'app', 'model_metrics.json')
        if not os.path.exists(metrics_path):
            raise HTTPException(404, "Metrics not found. Run train_pipeline.py first.")
        with open(metrics_path) as f:
            return json.load(f)
    
    # ── Dataset Stats ─────────────────────────────────────────
    @app.get("/stats")
    async def stats():
        """Return dataset statistics."""
        stats_path = os.path.join(os.path.dirname(__file__), 'app', 'dataset_stats.json')
        if not os.path.exists(stats_path):
            raise HTTPException(404, "Stats not found. Run train_pipeline.py first.")
        with open(stats_path) as f:
            return json.load(f)
    
    # ── Update Threshold ──────────────────────────────────────
    @app.post("/threshold")
    async def update_threshold(update: ThresholdUpdate):
        """Update the fraud decision threshold at runtime."""
        if simulator is None:
            raise HTTPException(503, "Model not loaded.")
        
        old_threshold = simulator.threshold
        simulator.set_threshold(update.threshold)
        
        return {
            'status': 'updated',
            'old_threshold': old_threshold,
            'new_threshold': simulator.threshold,
            'message': f"Threshold updated from {old_threshold:.2f} to {simulator.threshold:.2f}"
        }
    
    # ── Dashboard Stats ───────────────────────────────────────
    @app.get("/dashboard")
    async def dashboard():
        """Real-time dashboard statistics."""
        if simulator is None:
            raise HTTPException(503, "Model not loaded.")
        return simulator.get_dashboard_stats()
    
    return app


# ── Flask Fallback (if FastAPI unavailable) ─────────────────────

def create_flask_app():
    """Flask-based API as fallback."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise ImportError("Neither FastAPI nor Flask available.")
    
    flask_app = Flask(__name__)
    model = None
    simulator = None
    feature_cols = get_feature_columns()
    
    try:
        model = load_model('gradient_boosting')
        simulator = RealTimeSimulator(model, feature_cols)
    except Exception:
        pass
    
    @flask_app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'model_loaded': simulator is not None})
    
    @flask_app.route('/metrics', methods=['GET'])
    def metrics():
        try:
            with open('app/model_metrics.json') as f:
                return jsonify(json.load(f))
        except FileNotFoundError:
            return jsonify({'error': 'Run train_pipeline.py first'}), 404
    
    @flask_app.route('/predict', methods=['POST'])
    def predict():
        if simulator is None:
            return jsonify({'error': 'Model not loaded'}), 503
        data = request.json
        try:
            features = np.array([data.get(f, 0) for f in feature_cols], dtype=np.float32)
            result = simulator.predict_transaction(features)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return flask_app


if __name__ == '__main__':
    import uvicorn
    app = create_app()
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
