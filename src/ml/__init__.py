"""
ML 智能層模組

提供機器學習預測功能，包括：
- 特徵工程（技術指標、市場特徵）
- 模型訓練（Random Forest、XGBoost）
- 預測管道（訊號生成）
"""

from .features import FeatureEngineering
from .model import StockPredictor
from .predictor import SignalGenerator

__all__ = [
    'FeatureEngineering',
    'StockPredictor',
    'SignalGenerator'
]
