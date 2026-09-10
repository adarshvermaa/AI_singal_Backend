import logging
import os
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = logging.getLogger(__name__)

class XGBoostSignalModel:
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
        self.model = None
        self.is_loaded = False
        
        if HAS_XGB:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                objective="multi:softprob",
                num_class=3
            )
        else:
            logger.warning("xgboost not installed, using heuristic fallback")

    def train(self, X: np.ndarray, y: np.ndarray):
        if not HAS_XGB:
            logger.error("Cannot train: xgboost not available.")
            return
        self.model.fit(X, y)
        self.is_loaded = True

    def predict_proba(self, features: np.ndarray) -> Dict[str, float]:
        if not self.is_loaded or not HAS_XGB:
            return self._heuristic_fallback(features)
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        probs = self.model.predict_proba(features)[0]
        return {
            "BUY": float(probs[0]),
            "HOLD": float(probs[1]),
            "SELL": float(probs[2])
        }
        
    def _heuristic_fallback(self, features: np.ndarray) -> Dict[str, float]:
        if features.size == 0:
            return {"BUY": 0.33, "HOLD": 0.34, "SELL": 0.33}
            
        val = float(np.mean(features))
        if val > 0.5:
            return {"BUY": 0.6, "HOLD": 0.3, "SELL": 0.1}
        elif val < -0.5:
            return {"BUY": 0.1, "HOLD": 0.3, "SELL": 0.6}
        return {"BUY": 0.2, "HOLD": 0.6, "SELL": 0.2}

    def get_top_features(self, top_n: int = 10) -> List[Dict[str, Any]]:
        if not self.is_loaded or not HAS_XGB:
            return []
            
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        results = []
        for idx in indices:
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            results.append({
                "feature": name,
                "importance": float(importance[idx])
            })
        return results

    def save(self, path: str):
        if self.is_loaded and HAS_XGB:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save_model(path)

    def load(self, path: str):
        if HAS_XGB and os.path.exists(path):
            self.model.load_model(path)
            self.is_loaded = True
        else:
            logger.info(f"Model file {path} not found or xgboost missing. Using heuristic fallback.")
