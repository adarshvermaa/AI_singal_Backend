import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

class MLFeatureBuilder:
    def __init__(self, use_standard_scaler: bool = True):
        self.feature_names: List[str] = []
        self.scaler = None
        if HAS_SKLEARN:
            self.scaler = StandardScaler() if use_standard_scaler else MinMaxScaler()
        else:
            logger.warning("scikit-learn not installed, falling back to basic normalizer")
        
        self.is_fitted = False

    def build_features(self, candles: List[Dict[str, Any]], orderbook: Dict[str, Any], trade_flow: Dict[str, Any], futures: Dict[str, Any]) -> np.ndarray:
        # Example feature extraction logic
        features = []
        self.feature_names = []
        
        # Candles
        if candles:
            latest = candles[-1]
            features.extend([
                float(latest.get("close", 0.0)),
                float(latest.get("rsi", 50.0)),
                float(latest.get("macd", 0.0))
            ])
            self.feature_names.extend(["close", "rsi", "macd"])
            
        # Orderbook
        features.extend([
            float(orderbook.get("bid_ask_spread", 0.0)),
            float(orderbook.get("imbalance", 0.0))
        ])
        self.feature_names.extend(["ob_spread", "ob_imbalance"])
        
        # Trade Flow
        features.extend([
            float(trade_flow.get("buy_volume", 0.0)),
            float(trade_flow.get("sell_volume", 0.0))
        ])
        self.feature_names.extend(["tf_buy_vol", "tf_sell_vol"])
        
        # Futures
        features.extend([
            float(futures.get("funding_rate", 0.0)),
            float(futures.get("open_interest", 0.0))
        ])
        self.feature_names.extend(["fut_funding", "fut_oi"])
        
        return np.array(features, dtype=np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
            return X_scaled
        else:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0) + 1e-9
            self.is_fitted = True
            return (X - mean) / std

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return X
        if self.scaler:
            return self.scaler.transform(X)
        return X

    def format_for_lstm(self, X: np.ndarray, sequence_length: int = 60) -> np.ndarray:
        num_samples, num_features = X.shape
        if num_samples < sequence_length:
            pad_size = sequence_length - num_samples
            X = np.pad(X, ((pad_size, 0), (0, 0)), mode='edge')
            num_samples = sequence_length
            
        sequences = []
        for i in range(num_samples - sequence_length + 1):
            sequences.append(X[i:i+sequence_length])
            
        return np.array(sequences, dtype=np.float32)

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "scaler_type": type(self.scaler).__name__ if self.scaler else "basic"
        }
