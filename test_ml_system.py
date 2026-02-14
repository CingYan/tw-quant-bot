#!/usr/bin/env python3
"""
ML 智能層完整測試

測試項目：
1. 特徵工程
2. ML 模型訓練與評估
3. 訊號生成（技術分析 + ML）
"""

import pandas as pd
from pathlib import Path
from src.data.fetcher_v2 import TWSEDataFetcherV2
from src.ml.features import FeatureEngineering
from src.ml.model import StockPredictor
from src.ml.predictor import SignalGenerator


def test_ml_system():
    """完整 ML 系統測試"""
    
    print("=" * 60)
    print("🚀 台股量化交易系統 - ML 智能層測試")
    print("=" * 60)
    print()
    
    # ==================== 1. 資料載入 ====================
    print("📥 階段 1: 資料載入")
    print("-" * 60)
    
    fetcher = TWSEDataFetcherV2()
    data = fetcher.get_stock_data('2330.TW')
    df = pd.DataFrame(data)
    
    print(f"✅ 載入台積電資料: {len(df)} 筆")
    print(f"   日期範圍: {df['date'].min()} ~ {df['date'].max()}")
    print()
    
    # ==================== 2. 特徵工程 ====================
    print("🔧 階段 2: 特徵工程")
    print("-" * 60)
    
    fe = FeatureEngineering()
    df_features = fe.create_features(df)
    
    print(f"✅ 創建特徵: {len(df_features.columns)} 個")
    print(f"   有效資料: {len(df_features)} 筆")
    print(f"   目標分佈:")
    print(f"     - 漲: {(df_features['target'] == 1).sum()} 筆")
    print(f"     - 跌: {(df_features['target'] == 0).sum()} 筆")
    print()
    
    # ==================== 3. ML 模型訓練 ====================
    print("🎯 階段 3: ML 模型訓練")
    print("-" * 60)
    
    predictor = StockPredictor(model_type='random_forest')
    X_train, X_test, y_train, y_test = predictor.prepare_data(df_features)
    
    print(f"   訓練集: {len(X_train)} 筆")
    print(f"   測試集: {len(X_test)} 筆")
    print()
    
    metrics = predictor.train(X_train, y_train, X_test, y_test)
    print()
    
    # 特徵重要性
    print("📈 Top 10 重要特徵:")
    importance_df = predictor.get_feature_importance(top_n=10)
    for idx, row in importance_df.iterrows():
        print(f"   {idx+1}. {row['feature']:20s} {row['importance']:.4f}")
    print()
    
    # 儲存模型
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = 'models/stock_predictor_2330.pkl'
    predictor.save(model_path)
    print()
    
    # ==================== 4. 訊號生成 ====================
    print("📡 階段 4: 訊號生成（技術分析 + ML）")
    print("-" * 60)
    
    signal_gen = SignalGenerator(
        predictor=predictor,
        ml_threshold=0.6,
        use_technical=True,
        use_ml=True
    )
    
    df_signals = signal_gen.generate_signals(df)
    
    # 最新訊號
    latest_signal = signal_gen.get_latest_signal(df)
    
    print("最新交易訊號:")
    print(f"   日期: {latest_signal['date']}")
    print(f"   收盤價: {latest_signal['close']:.2f} TWD")
    print(f"   綜合訊號: {latest_signal['signal']}")
    print(f"   技術分析分數: {latest_signal['technical_score']}")
    print(f"   ML 上漲機率: {latest_signal['ml_proba_up']:.2%}")
    print(f"   RSI(14): {latest_signal['rsi_14']:.2f}")
    print(f"   MA5/MA20 金叉: {latest_signal['ma_5_20_cross']}")
    print(f"   推薦: {latest_signal['recommendation']}")
    print()
    
    # 近期訊號統計
    print("📊 近 30 天訊號統計:")
    recent_signals = df_signals['final_signal'].tail(30).value_counts().sort_index()
    for signal, count in recent_signals.items():
        signal_name = {-1: "賣出", 0: "觀望", 1: "買入"}.get(signal, "未知")
        print(f"   {signal_name:4s} ({signal:2d}): {count:2d} 天")
    print()
    
    # ==================== 5. 回測整合 ====================
    print("🔄 階段 5: 回測整合（簡易版）")
    print("-" * 60)
    
    # 統計訊號準確率（簡化版）
    df_valid = df_signals.dropna(subset=['target', 'final_signal'])
    
    buy_signals = df_valid[df_valid['final_signal'] == 1]
    sell_signals = df_valid[df_valid['final_signal'] == -1]
    
    if len(buy_signals) > 0:
        buy_accuracy = (buy_signals['target'] == 1).mean()
        print(f"   買入訊號準確率: {buy_accuracy:.2%} ({len(buy_signals)} 次)")
    
    if len(sell_signals) > 0:
        sell_accuracy = (sell_signals['target'] == 0).mean()
        print(f"   賣出訊號準確率: {sell_accuracy:.2%} ({len(sell_signals)} 次)")
    
    print()
    
    # ==================== 總結 ====================
    print("=" * 60)
    print("✅ ML 智能層測試完成！")
    print("=" * 60)
    print()
    print("完成項目:")
    print("  ✅ 特徵工程（46 個技術指標）")
    print("  ✅ Random Forest 模型訓練")
    print(f"  ✅ 測試集準確率: {metrics.get('test_accuracy', 0):.2%}")
    print("  ✅ 訊號生成（技術 + ML 雙重確認）")
    print(f"  ✅ 模型已儲存: {model_path}")
    print()
    print("下一步:")
    print("  📱 階段四: 監控介面（Telegram + Web UI）")
    print()


if __name__ == '__main__':
    test_ml_system()
