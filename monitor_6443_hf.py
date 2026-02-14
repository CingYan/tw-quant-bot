#!/usr/bin/env python3
"""
6443.TW 超高頻監控 - 每 1 分鐘
"""
import yfinance as yf
import json
import time
from datetime import datetime

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
    
    if rsi > 75 or ma5 < ma20:
        signal = 'SELL'
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
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 超高頻監控啟動 {symbol} (每60秒)")
    
    while True:
        try:
            now = datetime.now()
            hour = now.hour
            
            # 交易時段 09:00-13:30
            if hour < 9 or (hour == 13 and now.minute > 30) or hour > 13:
                if hour >= 14:
                    print(f"[{now.strftime('%H:%M:%S')}] 收盤")
                    data = get_signal(symbol)
                    with open(SIGNAL_FILE, 'w') as f:
                        f.write(f"CLOSED|{data['close']}|{data['rsi']:.1f}|{now.isoformat()}")
                    break
                time.sleep(30)
                continue
            
            data = get_signal(symbol)
            
            # 賣出檢查
            if data['signal'] == 'SELL' and last_action != 'SELL' and state.get('shares', 0) > 0:
                pnl = (data['close'] - state['entry_price']) * state['shares']
                pnl_pct = ((data['close'] / state['entry_price']) - 1) * 100
                
                print(f"🔴 SELL @ {data['close']:.2f} | P&L: {pnl:+,.0f} ({pnl_pct:+.2f}%)")
                with open(SIGNAL_FILE, 'w') as f:
                    f.write(f"SELL|{data['close']}|{pnl}|{pnl_pct:.2f}|{now.isoformat()}")
                
                state['shares'] = 0
                state['last_action'] = 'SELL'
                state['exit_price'] = data['close']
                save_state(state)
                last_action = 'SELL'
            
            # 買入檢查
            elif data['signal'] == 'BUY' and last_action != 'BUY' and state.get('shares', 0) == 0:
                cash = state.get('cash', 200000)
                shares = int(cash * 0.8 / data['close'] / 1000) * 1000
                
                if shares >= 1000:
                    print(f"🟢 BUY {shares} @ {data['close']:.2f}")
                    with open(SIGNAL_FILE, 'w') as f:
                        f.write(f"BUY|{data['close']}|{shares}|{now.isoformat()}")
                    
                    state['shares'] = shares
                    state['entry_price'] = data['close']
                    state['last_action'] = 'BUY'
                    save_state(state)
                    last_action = 'BUY'
            else:
                # 靜默日誌
                print(f"[{now.strftime('%H:%M:%S')}] {data['close']:.2f} | RSI {data['rsi']:.1f} | {data['signal']}")
            
            # 每 60 秒檢查
            time.sleep(60)
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ {e}")
            time.sleep(30)

if __name__ == '__main__':
    main()
