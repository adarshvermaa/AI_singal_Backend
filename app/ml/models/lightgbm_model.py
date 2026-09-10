import logging
import os
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logger = logging.getLogger(__name__)

class LightGBMSignalModel:
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
        self.model = None
        self.is_loaded = False
        
        if HAS_LGB:
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                objective="multiclass",
                num_class=3
            )
        else:
            logger.warning("lightgbm not installed, using heuristic fallback")

    def train(self, X: np.ndarray, y: np.ndarray):
        if not HAS_LGB:
            logger.error("Cannot train: lightgbm not available.")
            return
        self.model.fit(X, y)
        self.is_loaded = True

    def predict_proba(self, features: np.ndarray) -> Dict[str, float]:
        if not self.is_loaded or not HAS_LGB:
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
        if val > 0.3:
            return {"BUY": 0.5, "HOLD": 0.3, "SELL": 0.2}
        elif val < -0.3:
            return {"BUY": 0.2, "HOLD": 0.3, "SELL": 0.5}
        return {"BUY": 0.2, "HOLD": 0.6, "SELL": 0.2}

    def get_top_features(self, top_n: int = 10) -> List[Dict[str, Any]]:
        if not self.is_loaded or not HAS_LGB:
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
        import joblib
        if self.is_loaded and HAS_LGB:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)

    def load(self, path: str):
        import joblib
        if HAS_LGB and os.path.exists(path):
            self.model = joblib.load(path)
            self.is_loaded = True
        else:
            logger.info(f"Model file {path} not found or lightgbm missing. Using fallback.")
