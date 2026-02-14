"""
訊號生成模組

整合技術分析和 ML 預測，生成交易訊號。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .features import FeatureEngineering
from .model import StockPredictor


class SignalGenerator:
    """交易訊號生成器（技術分析 + ML 預測）"""
    
    def __init__(
        self,
        predictor: Optional[StockPredictor] = None,
        ml_threshold: float = 0.6,
        use_technical: bool = True,
        use_ml: bool = True
    ):
        """
        初始化訊號生成器
        
        Args:
            predictor: ML 預測模型
            ml_threshold: ML 預測機率門檻（> threshold 才產生訊號）
            use_technical: 是否使用技術分析訊號
            use_ml: 是否使用 ML 預測訊號
        """
        self.predictor = predictor
        self.ml_threshold = ml_threshold
        self.use_technical = use_technical
        self.use_ml = use_ml
        self.fe = FeatureEngineering()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易訊號
        
        Args:
            df: 包含 OHLCV 的原始資料
        
        Returns:
            包含訊號的 DataFrame
        """
        # 1. 特徵工程
        df_features = self.fe.create_features(df)
        
        # 2. 技術分析訊號
        if self.use_technical:
            df_features = self._add_technical_signals(df_features)
        
        # 3. ML 預測訊號
        if self.use_ml and self.predictor is not None:
            df_features = self._add_ml_signals(df_features)
        
        # 4. 綜合訊號
        df_features = self._combine_signals(df_features)
        
        return df_features
    
    def _add_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技術分析訊號"""
        # MA 交叉訊號
        df['signal_ma_cross'] = 0
        df.loc[df['ma_5_20_cross'] == 1, 'signal_ma_cross'] = 1  # 金叉買入
        df.loc[df['ma_5_20_cross'] == 0, 'signal_ma_cross'] = -1  # 死叉賣出
        
        # RSI 超買超賣訊號
        df['signal_rsi'] = 0
        df.loc[df['rsi_14'] < 30, 'signal_rsi'] = 1  # 超賣買入
        df.loc[df['rsi_14'] > 70, 'signal_rsi'] = -1  # 超買賣出
        
        # MACD 訊號
        df['signal_macd'] = 0
        df.loc[df['macd_hist'] > 0, 'signal_macd'] = 1  # MACD 金叉
        df.loc[df['macd_hist'] < 0, 'signal_macd'] = -1  # MACD 死叉
        
        # 布林通道訊號
        df['signal_bb'] = 0
        df.loc[df['bb_position'] < 0.2, 'signal_bb'] = 1  # 觸及下軌買入
        df.loc[df['bb_position'] > 0.8, 'signal_bb'] = -1  # 觸及上軌賣出
        
        # 技術分析綜合分數（-4 到 4）
        df['technical_score'] = (
            df['signal_ma_cross'] +
            df['signal_rsi'] +
            df['signal_macd'] +
            df['signal_bb']
        )
        
        return df
    
    def _add_ml_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 ML 預測訊號"""
        # 使用模型訓練時的特徵名稱
        if self.predictor.feature_names is None:
            raise ValueError("模型未訓練或未載入特徵名稱")
        
        # 確保 DataFrame 包含所有必要特徵
        missing_features = set(self.predictor.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"缺少特徵: {missing_features}")
        
        X = df[self.predictor.feature_names].values
        
        # 預測機率
        proba = self.predictor.predict_proba(X)
        df['ml_proba_up'] = proba[:, 1]  # 上漲機率
        
        # ML 訊號（基於機率門檻）
        df['signal_ml'] = 0
        df.loc[df['ml_proba_up'] > self.ml_threshold, 'signal_ml'] = 1  # 高機率上漲
        df.loc[df['ml_proba_up'] < (1 - self.ml_threshold), 'signal_ml'] = -1  # 高機率下跌
        
        return df
    
    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """綜合技術分析和 ML 訊號"""
        # 初始化綜合訊號
        df['final_signal'] = 0
        
        # 策略 1: 技術分析 + ML 雙重確認
        if self.use_technical and self.use_ml:
            # 技術分析強烈看多（>= 2）且 ML 看多
            df.loc[
                (df['technical_score'] >= 2) & (df['signal_ml'] == 1),
                'final_signal'
            ] = 1
            
            # 技術分析強烈看空（<= -2）且 ML 看空
            df.loc[
                (df['technical_score'] <= -2) & (df['signal_ml'] == -1),
                'final_signal'
            ] = -1
        
        # 策略 2: 僅使用技術分析
        elif self.use_technical:
            df.loc[df['technical_score'] >= 3, 'final_signal'] = 1
            df.loc[df['technical_score'] <= -3, 'final_signal'] = -1
        
        # 策略 3: 僅使用 ML
        elif self.use_ml:
            df['final_signal'] = df['signal_ml']
        
        return df
    
    def get_latest_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        獲取最新訊號
        
        Args:
            df: 原始 OHLCV 資料
        
        Returns:
            最新訊號字典
        """
        df_signals = self.generate_signals(df)
        latest = df_signals.iloc[-1]
        
        signal_dict = {
            'date': latest.name if hasattr(latest, 'name') else 'latest',
            'close': latest['close'],
            'signal': int(latest['final_signal']),
            'technical_score': int(latest.get('technical_score', 0)),
            'ml_proba_up': float(latest.get('ml_proba_up', 0.5)),
            'rsi_14': float(latest['rsi_14']),
            'ma_5_20_cross': bool(latest['ma_5_20_cross']),
            'recommendation': self._get_recommendation(latest)
        }
        
        return signal_dict
    
    @staticmethod
    def _get_recommendation(row: pd.Series) -> str:
        """根據訊號生成推薦"""
        signal = int(row['final_signal'])
        
        if signal == 1:
            return "🟢 買入訊號"
        elif signal == -1:
            return "🔴 賣出訊號"
        else:
            return "⚪ 觀望"


# ==================== 測試程式碼 ====================

if __name__ == '__main__':
    from ..data.fetcher_v2 import TWSEDataFetcherV2
    import pandas as pd
    
    print("🚀 開始訊號生成測試\n")
    
    # 1. 載入資料
    print("📥 載入台積電資料...")
    fetcher = TWSEDataFetcherV2()
    data = fetcher.get_stock_data('2330.TW')
    df = pd.DataFrame(data)
    print(f"✅ 載入 {len(df)} 筆資料\n")
    
    # 2. 載入 ML 模型（如果有）
    try:
        predictor = StockPredictor.load('models/stock_predictor_2330.pkl')
        print("✅ ML 模型已載入\n")
        use_ml = True
    except:
        print("⚠️ ML 模型未找到，僅使用技術分析\n")
        predictor = None
        use_ml = False
    
    # 3. 創建訊號生成器
    signal_gen = SignalGenerator(
        predictor=predictor,
        ml_threshold=0.6,
        use_technical=True,
        use_ml=use_ml
    )
    
    # 4. 生成訊號
    print("🔧 生成交易訊號...")
    df_signals = signal_gen.generate_signals(df)
    print(f"✅ 完成\n")
    
    # 5. 最新訊號
    latest_signal = signal_gen.get_latest_signal(df)
    
    print("📊 最新交易訊號:")
    print(f"  日期: {latest_signal['date']}")
    print(f"  收盤價: {latest_signal['close']:.2f}")
    print(f"  綜合訊號: {latest_signal['signal']}")
    print(f"  技術分析分數: {latest_signal['technical_score']}")
    if use_ml:
        print(f"  ML 上漲機率: {latest_signal['ml_proba_up']:.2%}")
    print(f"  RSI(14): {latest_signal['rsi_14']:.2f}")
    print(f"  MA5/MA20 金叉: {latest_signal['ma_5_20_cross']}")
    print(f"\n  推薦: {latest_signal['recommendation']}")
    
    # 6. 近期訊號統計
    print("\n📈 近 20 天訊號分佈:")
    recent_signals = df_signals['final_signal'].tail(20).value_counts()
    print(recent_signals)
    
    print("\n✅ 訊號生成測試完成！")
