"""
ML 模型訓練模組

提供股價預測模型的訓練、評估和預測功能。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import json
from pathlib import Path


class StockPredictor:
    """股價預測模型（二元分類：漲/跌）"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        初始化預測模型
        
        Args:
            model_type: 模型類型 ('random_forest', 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.metrics = {}
        
        # 初始化模型
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"不支援的模型類型: {model_type}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        準備訓練資料
        
        Args:
            df: 包含特徵和目標的 DataFrame
            target_col: 目標變數欄位名稱
            test_size: 測試集比例
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # 排除非特徵欄位
        exclude_cols = [target_col, 'future_return', 'target_3class', 'date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 提取特徵和目標
        X = df[feature_cols].values
        y = df[target_col].values
        
        # 儲存特徵名稱
        self.feature_names = feature_cols
        
        # 時間序列分割（避免未來資訊洩漏）
        # 使用最後 test_size 的資料作為測試集
        split_idx = int(len(df) * (1 - test_size))
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        訓練模型
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練目標
            X_test: 測試特徵（可選）
            y_test: 測試目標（可選）
        
        Returns:
            評估指標字典
        """
        print("🎯 開始訓練模型...")
        print(f"訓練樣本數: {len(X_train)}")
        
        # 訓練
        self.model.fit(X_train, y_train)
        
        # 評估
        metrics = {}
        
        # 訓練集評估
        y_train_pred = self.model.predict(X_train)
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        
        # 測試集評估
        if X_test is not None and y_test is not None:
            y_test_pred = self.model.predict(X_test)
            metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            metrics['test_precision'] = precision_score(y_test, y_test_pred, zero_division=0)
            metrics['test_recall'] = recall_score(y_test, y_test_pred, zero_division=0)
            metrics['test_f1'] = f1_score(y_test, y_test_pred, zero_division=0)
            
            print(f"\n📊 訓練集準確率: {metrics['train_accuracy']:.4f}")
            print(f"📊 測試集準確率: {metrics['test_accuracy']:.4f}")
            print(f"📊 精確率: {metrics['test_precision']:.4f}")
            print(f"📊 召回率: {metrics['test_recall']:.4f}")
            print(f"📊 F1 分數: {metrics['test_f1']:.4f}")
            
            print(f"\n分類報告:")
            print(classification_report(y_test, y_test_pred, target_names=['跌', '漲']))
        
        self.metrics = metrics
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測
        
        Args:
            X: 特徵矩陣
        
        Returns:
            預測結果（0: 跌, 1: 漲）
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先調用 train()")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        預測機率
        
        Args:
            X: 特徵矩陣
        
        Returns:
            預測機率 (N x 2)，第二列為上漲機率
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先調用 train()")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        獲取特徵重要性
        
        Args:
            top_n: 返回前 N 個重要特徵
        
        Returns:
            特徵重要性 DataFrame
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("模型尚未訓練")
        
        importance = self.model.feature_importances_
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance.head(top_n)
    
    def save(self, filepath: str):
        """儲存模型"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ 模型已儲存至: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StockPredictor':
        """載入模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.feature_names = model_data['feature_names']
        predictor.metrics = model_data['metrics']
        
        print(f"✅ 模型已載入: {filepath}")
        return predictor


# ==================== 測試程式碼 ====================

if __name__ == '__main__':
    from ..data.fetcher_v2 import TWSEDataFetcherV2
    from .features import FeatureEngineering
    
    print("🚀 開始 ML 模型訓練測試\n")
    
    # 1. 載入資料
    print("📥 載入台積電資料...")
    fetcher = TWSEDataFetcherV2()
    data = fetcher.get_stock_data('2330.TW')
    df = pd.DataFrame(data)
    print(f"✅ 載入 {len(df)} 筆資料\n")
    
    # 2. 特徵工程
    print("🔧 開始特徵工程...")
    fe = FeatureEngineering()
    df_features = fe.create_features(df)
    print(f"✅ 創建 {len(df_features.columns)} 個特徵\n")
    
    # 3. 準備訓練資料
    print("📊 準備訓練資料...")
    predictor = StockPredictor(model_type='random_forest')
    X_train, X_test, y_train, y_test = predictor.prepare_data(df_features)
    print(f"✅ 訓練集: {len(X_train)} 筆")
    print(f"✅ 測試集: {len(X_test)} 筆\n")
    
    # 4. 訓練模型
    metrics = predictor.train(X_train, y_train, X_test, y_test)
    
    # 5. 特徵重要性
    print("\n📈 Top 10 重要特徵:")
    importance_df = predictor.get_feature_importance(top_n=10)
    print(importance_df.to_string(index=False))
    
    # 6. 儲存模型
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    predictor.save(f'models/stock_predictor_2330.pkl')
    
    print("\n✅ ML 模型訓練完成！")
