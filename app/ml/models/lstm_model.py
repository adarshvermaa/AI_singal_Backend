import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        Module = object

logger = logging.getLogger(__name__)

if HAS_TORCH:
    class BiLSTMAttentionModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 3):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, bidirectional=True
            )
            self.attention_weights = nn.Linear(hidden_dim * 2, 1)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
            
        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            lstm_out, _ = self.lstm(x)
            
            attn_scores = self.attention_weights(lstm_out)
            attn_weights = F.softmax(attn_scores, dim=1)
            
            context = torch.sum(attn_weights * lstm_out, dim=1)
            
            out = self.fc(context)
            return out, attn_weights

class LSTMSignalPredictor:
    def __init__(self, input_dim: int = 10, sequence_length: int = 60):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.model = None
        self.is_loaded = False
        
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = BiLSTMAttentionModel(input_dim=input_dim).to(self.device)
            self.model.eval()
        else:
            logger.warning("PyTorch not installed, using heuristic fallback")

    def predict_proba(self, sequence_features: np.ndarray) -> Dict[str, float]:
        if not self.is_loaded or not HAS_TORCH:
            return self._heuristic_fallback(sequence_features)
            
        if sequence_features.ndim == 2:
            sequence_features = np.expand_dims(sequence_features, axis=0)
            
        with torch.no_grad():
            x_tensor = torch.tensor(sequence_features, dtype=torch.float32).to(self.device)
            logits, attn = self.model(x_tensor)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            
        return {
            "BUY": float(probs[0]),
            "HOLD": float(probs[1]),
            "SELL": float(probs[2])
        }

    def _heuristic_fallback(self, sequence_features: np.ndarray) -> Dict[str, float]:
        if sequence_features.size == 0:
            return {"BUY": 0.33, "HOLD": 0.34, "SELL": 0.33}
            
        last_step = np.mean(sequence_features[-1]) if sequence_features.ndim == 2 else np.mean(sequence_features[0, -1])
        first_step = np.mean(sequence_features[0]) if sequence_features.ndim == 2 else np.mean(sequence_features[0, 0])
        
        trend = last_step - first_step
        if trend > 0.2:
            return {"BUY": 0.55, "HOLD": 0.3, "SELL": 0.15}
        elif trend < -0.2:
            return {"BUY": 0.15, "HOLD": 0.3, "SELL": 0.55}
        return {"BUY": 0.25, "HOLD": 0.5, "SELL": 0.25}

    def save(self, path: str):
        if self.is_loaded and HAS_TORCH:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        if HAS_TORCH and os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.is_loaded = True
        else:
            logger.info(f"Model file {path} not found or torch missing. Using fallback.")
