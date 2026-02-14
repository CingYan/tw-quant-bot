"""
Web UI 儀表板

提供：
- 即時價格監控
- 訊號歷史記錄
- 績效圖表
- 模型指標
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import plotly
import plotly.graph_objs as go
import json
from datetime import datetime
from pathlib import Path


app = Flask(__name__, template_folder='../../templates')


# ==================== API 端點 ====================

@app.route('/')
def index():
    """首頁"""
    return render_template('dashboard.html')


@app.route('/api/latest_signal')
def api_latest_signal():
    """獲取最新訊號"""
    try:
        from ..data.fetcher_v2 import TWSEDataFetcherV2
        from ..ml.predictor import SignalGenerator, StockPredictor
        
        # 載入資料
        fetcher = TWSEDataFetcherV2()
        data = fetcher.get_stock_data('2330.TW')
        df = pd.DataFrame(data)
        
        # 載入模型
        predictor = StockPredictor.load('models/stock_predictor_2330.pkl')
        
        # 訊號生成
        signal_gen = SignalGenerator(predictor=predictor, ml_threshold=0.6)
        latest_signal = signal_gen.get_latest_signal(df)
        
        # 轉換為 JSON 可序列化格式
        result = {
            'date': str(latest_signal['date']),
            'close': float(latest_signal['close']),
            'signal': int(latest_signal['signal']),
            'technical_score': int(latest_signal['technical_score']),
            'ml_proba_up': float(latest_signal['ml_proba_up']),
            'rsi_14': float(latest_signal['rsi_14']),
            'ma_5_20_cross': bool(latest_signal['ma_5_20_cross']),
            'recommendation': latest_signal['recommendation']
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/price_chart')
def api_price_chart():
    """價格走勢圖"""
    try:
        from ..data.fetcher_v2 import TWSEDataFetcherV2
        
        fetcher = TWSEDataFetcherV2()
        data = fetcher.get_stock_data('2330.TW')
        df = pd.DataFrame(data)
        
        # 只取最近 90 天
        df = df.tail(90)
        
        # 創建 K 線圖
        fig = go.Figure(data=[
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'
            )
        ])
        
        fig.update_layout(
            title='台積電 2330.TW - 90 天走勢',
            xaxis_title='日期',
            yaxis_title='價格 (TWD)',
            template='plotly_dark',
            height=400
        )
        
        return jsonify(json.loads(fig.to_json()))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/signal_history')
def api_signal_history():
    """訊號歷史記錄"""
    try:
        from ..data.fetcher_v2 import TWSEDataFetcherV2
        from ..ml.predictor import SignalGenerator, StockPredictor
        
        # 載入資料
        fetcher = TWSEDataFetcherV2()
        data = fetcher.get_stock_data('2330.TW')
        df = pd.DataFrame(data)
        
        # 載入模型
        predictor = StockPredictor.load('models/stock_predictor_2330.pkl')
        
        # 訊號生成
        signal_gen = SignalGenerator(predictor=predictor, ml_threshold=0.6)
        df_signals = signal_gen.generate_signals(df)
        
        # 只取有訊號的記錄
        df_filtered = df_signals[df_signals['final_signal'] != 0].tail(20)
        
        # 轉換為 JSON
        result = []
        for _, row in df_filtered.iterrows():
            result.append({
                'date': str(row.name) if hasattr(row, 'name') else 'N/A',
                'close': float(row['close']),
                'signal': int(row['final_signal']),
                'technical_score': int(row['technical_score']),
                'ml_proba_up': float(row['ml_proba_up'])
            })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_metrics')
def api_model_metrics():
    """模型評估指標"""
    try:
        from ..ml.model import StockPredictor
        
        predictor = StockPredictor.load('models/stock_predictor_2330.pkl')
        
        return jsonify(predictor.metrics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feature_importance')
def api_feature_importance():
    """特徵重要性"""
    try:
        from ..ml.model import StockPredictor
        
        predictor = StockPredictor.load('models/stock_predictor_2330.pkl')
        importance_df = predictor.get_feature_importance(top_n=15)
        
        # 轉換為 Plotly 圖表
        fig = go.Figure(data=[
            go.Bar(
                x=importance_df['importance'].tolist()[::-1],
                y=importance_df['feature'].tolist()[::-1],
                orientation='h',
                marker=dict(color='lightblue')
            )
        ])
        
        fig.update_layout(
            title='Top 15 重要特徵',
            xaxis_title='重要性',
            yaxis_title='特徵',
            template='plotly_dark',
            height=500
        )
        
        return jsonify(json.loads(fig.to_json()))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== 啟動服務 ====================

def start_web_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """
    啟動 Web 服務器
    
    Args:
        host: 主機位址
        port: 埠號
        debug: 除錯模式
    """
    print("=" * 60)
    print("🌐 台股量化交易系統 - Web UI 儀表板")
    print("=" * 60)
    print(f"📍 服務位址：http://{host}:{port}")
    print("📊 功能：")
    print("  • 即時價格監控")
    print("  • 訊號歷史記錄")
    print("  • 績效圖表")
    print("  • 模型指標")
    print("=" * 60)
    print("\n按 Ctrl+C 停止服務器...\n")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    start_web_server(debug=True)
