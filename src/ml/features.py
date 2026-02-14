"""
特徵工程模組

從原始價格資料中提取技術指標和市場特徵，用於機器學習模型訓練。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class FeatureEngineering:
    """特徵工程類別，提取技術指標和市場特徵"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        從 OHLCV 資料創建完整特徵集
        
        Args:
            df: 包含 open, high, low, close, volume 的 DataFrame
        
        Returns:
            包含所有特徵的 DataFrame
        """
        df = df.copy()
        
        # 0. 資料前處理
        # 只保留必要欄位
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in required_cols if col in df.columns]].copy()
        
        # 轉換日期為 datetime 並設為 index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # 確保數值類型正確
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除全 NaN 的行
        df = df.dropna(how='all')
        
        # 1. 價格特徵
        df = self._add_price_features(df)
        
        # 2. 移動平均線特徵
        df = self._add_ma_features(df)
        
        # 3. 動量指標
        df = self._add_momentum_features(df)
        
        # 4. 波動率指標
        df = self._add_volatility_features(df)
        
        # 5. 成交量特徵
        df = self._add_volume_features(df)
        
        # 6. 趨勢指標
        df = self._add_trend_features(df)
        
        # 7. 目標變數（未來報酬）
        df = self._add_target(df)
        
        # 移除 NaN（前期無法計算的指標）
        df = df.dropna()
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加價格相關特徵"""
        # 日內波動率
        df['intraday_range'] = (df['high'] - df['low']) / df['close']
        
        # 開盤跳空（Gap）
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # 收盤相對位置（Close Position）
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def _add_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加移動平均線特徵"""
        # 多週期 MA
        for period in [5, 10, 20, 60]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'close_to_ma_{period}'] = (df['close'] - df[f'ma_{period}']) / df[f'ma_{period}']
        
        # MA 斜率（趨勢強度）
        df['ma_5_slope'] = df['ma_5'].pct_change(5)
        df['ma_20_slope'] = df['ma_20'].pct_change(5)
        
        # MA 交叉
        df['ma_5_20_cross'] = (df['ma_5'] > df['ma_20']).astype(int)
        df['ma_10_60_cross'] = (df['ma_10'] > df['ma_60']).astype(int)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加動量指標"""
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_28'] = self._calculate_rsi(df['close'], 28)
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波動率指標"""
        # 歷史波動率（標準差）
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std()
        
        # ATR (Average True Range)
        df['atr_14'] = self._calculate_atr(df, 14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量特徵"""
        # 成交量移動平均
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        # 成交量比率
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # OBV (On-Balance Volume)
        df['obv'] = self._calculate_obv(df)
        
        # 成交量趨勢
        df['volume_trend'] = df['volume_ma_5'] / df['volume_ma_20']
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趨勢指標"""
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # ADX (Average Directional Index)
        df['adx_14'] = self._calculate_adx(df, 14)
        
        return df
    
    def _add_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        添加目標變數（未來報酬）
        
        Args:
            df: 價格資料
            horizon: 預測時間範圍（天數）
        
        Returns:
            包含目標變數的 DataFrame
        """
        # 未來 N 天報酬
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # 二元分類目標（漲/跌）
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # 三元分類目標（大漲/持平/大跌）
        df['target_3class'] = pd.cut(
            df['future_return'],
            bins=[-np.inf, -0.02, 0.02, np.inf],
            labels=[0, 1, 2]  # 0: 跌, 1: 持平, 2: 漲
        ).astype(float)
        
        return df
    
    # ==================== 技術指標計算函數 ====================
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """計算 RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """計算 ATR (Average True Range)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2):
        """計算布林通道"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def _calculate_obv(df: pd.DataFrame) -> pd.Series:
        """計算 OBV (On-Balance Volume)"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """計算 MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """計算 ADX (Average Directional Index)"""
        # +DM 和 -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # ATR
        atr = FeatureEngineering._calculate_atr(df, period)
        
        # +DI 和 -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # DX 和 ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def get_feature_importance_names(self) -> List[str]:
        """返回特徵名稱列表（用於模型訓練）"""
        return [col for col in self.feature_names if col not in ['target', 'future_return', 'target_3class']]


# ==================== 測試程式碼 ====================

if __name__ == '__main__':
    # 測試特徵工程
    from ..data.fetcher_v2 import TWSEDataFetcherV2
    
    # 載入資料
    fetcher = TWSEDataFetcherV2()
    data = fetcher.get_stock_data('2330.TW')
    df = pd.DataFrame(data)
    
    # 創建特徵
    fe = FeatureEngineering()
    df_features = fe.create_features(df)
    
    print("✅ 特徵工程完成")
    print(f"📊 原始資料: {len(df)} 筆")
    print(f"📊 處理後資料: {len(df_features)} 筆")
    print(f"📊 特徵數量: {len(df_features.columns)} 個")
    print(f"\n特徵列表:")
    for i, col in enumerate(df_features.columns, 1):
        print(f"  {i}. {col}")
    
    # 檢查目標分佈
    print(f"\n目標變數分佈:")
    print(df_features['target'].value_counts())
    print(f"\n最新特徵（前 5 筆）:")
    print(df_features.tail(5)[['close', 'rsi_14', 'ma_5_20_cross', 'target']])
