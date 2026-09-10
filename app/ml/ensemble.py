import logging
from typing import Dict, Any, List, Optional
import numpy as np

from app.ml.models.xgboost_model import XGBoostSignalModel
from app.ml.models.lightgbm_model import LightGBMSignalModel
from app.ml.models.lstm_model import LSTMSignalPredictor

logger = logging.getLogger(__name__)

class SignalEnsemble:
    def __init__(self, xgb_model: XGBoostSignalModel, lgb_model: LightGBMSignalModel, lstm_model: LSTMSignalPredictor):
        self.xgb = xgb_model
        self.lgb = lgb_model
        self.lstm = lstm_model
        
        self.weights = {
            "xgb": 0.4,
            "lgb": 0.3,
            "lstm": 0.3
        }

    def predict(self, flat_features: np.ndarray, sequence_features: np.ndarray) -> Dict[str, Any]:
        xgb_preds = self.xgb.predict_proba(flat_features)
        lgb_preds = self.lgb.predict_proba(flat_features)
        lstm_preds = self.lstm.predict_proba(sequence_features)
        
        ensemble_probs = {}
        for cls in ["BUY", "HOLD", "SELL"]:
            ensemble_probs[cls] = (
                xgb_preds[cls] * self.weights["xgb"] +
                lgb_preds[cls] * self.weights["lgb"] +
                lstm_preds[cls] * self.weights["lstm"]
            )
            
        xgb_vote = max(xgb_preds, key=xgb_preds.get)
        lgb_vote = max(lgb_preds, key=lgb_preds.get)
        lstm_vote = max(lstm_preds, key=lstm_preds.get)
        
        votes = [xgb_vote, lgb_vote, lstm_vote]
        final_signal = max(ensemble_probs, key=ensemble_probs.get)
        
        if votes.count(final_signal) == 3:
            agreement = "UNANIMOUS"
        elif votes.count(final_signal) == 2:
            agreement = "MAJORITY"
        else:
            agreement = "SPLIT"
            
        top_features = self.xgb.get_top_features(top_n=5)
        reasons = [f["feature"] for f in top_features] if top_features else ["Trend alignment", "Momentum shift"]
            
        return {
            "signal": final_signal,
            "agreement": agreement,
            "probabilities": ensemble_probs,
            "model_votes": {
                "xgb": xgb_vote,
                "lgb": lgb_vote,
                "lstm": lstm_vote
            },
            "top_reasons": reasons
        }
