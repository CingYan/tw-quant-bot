#!/usr/bin/env python3
"""
6443.TW 高頻監控
每 5 分鐘檢查，只在有操作時輸出
"""
import yfinance as yf
import json
import time
from datetime import datetime
import os

STATE_FILE = '/home/node/clawd/tw-quant-bot/bot_state.json'
SIGNAL_FILE = '/home/node/clawd/tw-quant-bot/signal_output.txt'

def load_state():
    with open(STATE_FILE, 'r') as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def get_signal(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='3mo')
    close = float(df['Close'].iloc[-1])
    
    ma5 = float(df['Close'].rolling(5).mean().iloc[-1])
    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])
    
    # 賣出條件：RSI > 75 或 死叉
    if rsi > 75 or ma5 < ma20:
        signal = 'SELL'
    # 買入條件：金叉 + RSI < 65
    elif ma5 > ma20 and rsi < 65:
        signal = 'BUY'
    else:
        signal = 'HOLD'
    
    return {
        'signal': signal,
        'close': close,
        'rsi': rsi,
        'ma5': ma5,
        'ma20': ma20,
        'ma_cross': 'golden' if ma5 > ma20 else 'death'
    }

def main():
    state = load_state()
    symbol = state['symbol']
    last_action = state.get('last_action', 'BUY')
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 開始監控 {symbol}...")
    
    check_count = 0
    
    while True:
        try:
            now = datetime.now()
            hour = now.hour
            
            # 只在交易時段檢查 (09:00-13:30)
            if hour < 9 or (hour == 13 and now.minute > 30) or hour > 13:
                if hour >= 14:
                    print(f"[{now.strftime('%H:%M:%S')}] 收盤，停止監控")
                    # 寫入最終狀態
                    data = get_signal(symbol)
                    with open(SIGNAL_FILE, 'w') as f:
                        f.write(f"CLOSED|{data['close']}|{data['rsi']:.1f}|{now.isoformat()}")
                    break
                time.sleep(60)
                continue
            
            data = get_signal(symbol)
            check_count += 1
            
            # 檢查是否需要操作
            action_needed = False
            
            if data['signal'] == 'SELL' and last_action != 'SELL' and state.get('shares', 0) > 0:
                # 賣出訊號
                action_needed = True
                pnl = (data['close'] - state['entry_price']) * state['shares']
                pnl_pct = ((data['close'] / state['entry_price']) - 1) * 100
                
                msg = f"""🔴 **賣出訊號！**

📊 {symbol}
💵 價格：{data['close']:.2f} TWD
📈 RSI：{data['rsi']:.1f}
📊 MA：{'死叉' if data['ma_cross'] == 'death' else '金叉'}

⚡ 賣出 {state['shares']:,} 股
💰 P&L：{pnl:+,.0f} TWD ({pnl_pct:+.2f}%)
"""
                print(msg)
                with open(SIGNAL_FILE, 'w') as f:
                    f.write(f"SELL|{data['close']}|{pnl}|{pnl_pct:.2f}|{now.isoformat()}")
                
                state['shares'] = 0
                state['last_action'] = 'SELL'
                state['exit_price'] = data['close']
                state['exit_time'] = now.isoformat()
                save_state(state)
                last_action = 'SELL'
            
            elif data['signal'] == 'BUY' and last_action != 'BUY' and state.get('shares', 0) == 0:
                # 買入訊號（如果之前賣出了）
                action_needed = True
                cash = state.get('cash', 200000)
                shares = int(cash * 0.8 / data['close'] / 1000) * 1000
                
                if shares >= 1000:
                    msg = f"""🟢 **買入訊號！**

📊 {symbol}
💵 價格：{data['close']:.2f} TWD
📈 RSI：{data['rsi']:.1f}
📊 MA：金叉

⚡ 買入 {shares:,} 股
"""
                    print(msg)
                    with open(SIGNAL_FILE, 'w') as f:
                        f.write(f"BUY|{data['close']}|{shares}|{now.isoformat()}")
                    
                    state['shares'] = shares
                    state['entry_price'] = data['close']
                    state['last_action'] = 'BUY'
                    save_state(state)
                    last_action = 'BUY'
            
            # 每 10 次檢查輸出一次狀態（靜默模式）
            if check_count % 10 == 0:
                print(f"[{now.strftime('%H:%M:%S')}] #{check_count} | {data['close']:.2f} | RSI {data['rsi']:.1f} | {data['signal']}")
            
            # 5 分鐘檢查一次
            time.sleep(300)
        
        except KeyboardInterrupt:
            print("\n⏹️ 監控已停止")
            break
        except Exception as e:
            print(f"⚠️ 錯誤：{e}")
            time.sleep(60)

if __name__ == '__main__':
    main()
