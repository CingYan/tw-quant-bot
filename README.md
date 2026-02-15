# 台股當沖模擬交易 Bot - 使用指南

## 📋 概述

這是一個自動化的台股當沖模擬交易 bot，使用 Yahoo Finance 數據源，結合多個技術指標進行自動交易。

**主要特性：**
- ✅ 實時監控台股候選股清單
- ✅ 多指標聯動交易訊號（均線 + RSI + 成交量）
- ✅ 智能止損/止利管理
- ✅ 完整的交易紀錄和日誌
- ✅ Telegram 通知支持（透過 alert 檔案）

---

## 🚀 快速開始

### 1️⃣ 環境準備

**系統要求：**
- Python 3.8+
- Linux/Mac/Windows

**安裝依賴：**
```bash
# 方法一：使用 pip（推薦）
pip install yfinance pandas ta

# 方法二：使用 pip3
pip3 install yfinance pandas ta

# 方法三：使用 apt（Debian/Ubuntu）
sudo apt-get install python3-pip
pip3 install yfinance pandas ta

# 方法四：使用 conda
conda install -c conda-forge yfinance pandas ta-lib
```

### 2️⃣ 配置候選股清單

編輯 `/home/node/clawd/memory/daytrade-picks.md`，在檔案中加入要追蹤的台股代碼：

```markdown
# 當日候選股清單

- `2330.TW` (台積電)
- `2454.TW` (聯發科)
- `1303.TW` (南亞科)
```

**台股代碼格式：** `XXXX.TW`

### 3️⃣ 啟動 Bot

```bash
# 方式一：直接運行
python3 /home/node/clawd/scripts/daytrade-bot.py

# 方式二：後台運行（使用 nohup）
nohup python3 /home/node/clawd/scripts/daytrade-bot.py > /tmp/daytrade.log 2>&1 &

# 方式三：使用 screen（可恢復）
screen -S daytrade -d -m python3 /home/node/clawd/scripts/daytrade-bot.py

# 方式四：使用 tmux
tmux new-session -d -s daytrade "python3 /home/node/clawd/scripts/daytrade-bot.py"
```

### 4️⃣ 停止 Bot

```bash
# 前台運行：按 Ctrl+C

# 後台運行：
pkill -f daytrade-bot.py

# screen 會話：
screen -S daytrade -X quit

# tmux 會話：
tmux kill-session -t daytrade
```

---

## 📁 檔案結構

```
/home/node/clawd/
├── scripts/
│   ├── daytrade-bot.py           # 主程式
│   └── daytrade-bot.log          # 運行日誌
└── memory/
    ├── daytrade-picks.md         # 候選股清單（輸入）
    ├── daytrade-positions.md     # 當前持倉（實時更新）
    ├── daytrade-results.md       # 交易結果（累積）
    └── daytrade-alert.md         # 交易警報（供 Telegram 通知）
```

---

## ⚙️ 交易參數配置

編輯 `daytrade-bot.py` 中的 `CONFIG` 字典來自定義參數：

```python
CONFIG = {
    # 交易時段（台灣時間）
    'START_HOUR': 9,              # 9:00 開始
    'END_HOUR': 13,
    'END_MINUTE': 30,             # 13:30 結束
    'NO_OPEN_AFTER_HOUR': 13,     # 13:00 後不開新倉
    'FORCE_CLOSE_HOUR': 13,
    'FORCE_CLOSE_MINUTE': 25,     # 13:25 強制平倉
    
    # 風險管理
    'STOP_LOSS': -0.02,           # 停損 -2%
    'TAKE_PROFIT': 0.04,          # 停利 +4%
    
    # 檢查間隔
    'CHECK_INTERVAL': 60,         # 每 60 秒檢查一次
    
    # 技術指標
    'VOLUME_MULTIPLIER': 1.5,     # 成交量倍數 1.5x
}
```

---

## 🔧 交易策略詳解

### 買入訊號（三項條件同時滿足）

1. **均線突破**
   - 條件：5MA > 10MA（短期均線在中期均線上方）
   - 說明：表示短期上升趨勢

2. **RSI 動能**
   - 條件：RSI < 30（超賣）並反彈 OR RSI > 50（強勢）
   - 說明：結合價格變動判斷動能

3. **量能爆發**
   - 條件：成交量 > 前5根K棒平均的 1.5 倍
   - 說明：大量確認入場信號

### 賣出訊號（任一條件滿足）

1. **停損**
   - 條件：虧損 ≥ -2%
   - 目的：控制風險

2. **停利**
   - 條件：獲利 ≥ +4%
   - 目的：鎖定利潤

3. **時間停損**
   - 條件：13:25 強制平倉
   - 目的：當沖必須平倉，避免隔夜風險

---

## 📊 檔案說明

### `daytrade-picks.md` - 候選股清單
**用途：** 提供今日要監控的股票清單  
**來源：** 手動編輯或 8:50 cron 產生  
**格式：** Markdown，代碼格式 `XXXX.TW`  

**範例：**
```markdown
# 當日候選股清單

- `2330.TW` (台積電)
- `2454.TW` (聯發科)
```

### `daytrade-positions.md` - 當前持倉
**用途：** 記錄當前持倉的股票和進場價  
**更新：** 買入時更新，賣出時移除  
**格式：** JSON  

**範例：**
```json
{
  "2330.TW": {
    "entry_price": 155.50,
    "entry_time": "2026-02-10 09:30:00",
    "signal": "5MA > 10MA + 量能爆發"
  }
}
```

### `daytrade-results.md` - 交易結果
**用途：** 記錄所有已結清的交易  
**更新：** 每次賣出時追加一行  
**格式：** Markdown 表格  

**範例：**
```
| 股票代碼 | 進場價 | 出場價 | 淨利 | 淨利率 | 進場時間 | 出場時間 | 出場原因 |
|---------|--------|--------|------|--------|---------|---------|---------|
| 2330.TW | 155.50 | 156.70 | 1.20 | +0.77% | 09:30:00 | 10:45:00 | 停利 (+4%) |
```

### `daytrade-alert.md` - 交易警報
**用途：** 記錄每次交易行動的簡短訊息  
**更新：** 買入/賣出/停損停利時寫入  
**格式：** 純文本，每行一條警報  
**用途者：** 主 agent 讀取此檔案並通過 Telegram 發送通知  

**範例：**
```
[09:30:00] 【買入】2330.TW @ 155.50 | 5MA > 10MA + 量能爆發
[10:45:00] 【賣出】2330.TW @ 156.70 | 停利 (+4%) | 損益: +0.77%
[13:25:00] 【賣出】2454.TW @ 89.20 | 強制平倉 (13:25 截止) | 損益: -0.50%
```

---

## 🔍 監控和調試

### 查看實時日誌
```bash
tail -f /home/node/clawd/scripts/daytrade-bot.log
```

### 檢查當前持倉
```bash
cat /home/node/clawd/memory/daytrade-positions.md
```

### 檢查交易警報
```bash
cat /home/node/clawd/memory/daytrade-alert.md
```

### 查看交易結果統計
```bash
cat /home/node/clawd/memory/daytrade-results.md
```

---

## ⚠️ 注意事項

1. **數據延遲**
   - Yahoo Finance 有 15-20 分鐘延遲，實際交易時使用實時數據平台

2. **模擬交易**
   - 本 bot 為模擬交易，不涉及真實資金，用於策略驗證

3. **市場波動**
   - 台股在節假日或開盤前後可能無法取得數據

4. **時區**
   - 所有時間為台灣時間（Asia/Taipei, UTC+8）

5. **日誌大小**
   - 長期運行會產生大日誌檔，建議定期清理

---

## 🐛 常見問題

### Q: 沒有買入信號？
**A:** 檢查：
1. 候選股清單是否設定
2. 當前時間是否在 9:00-13:00
3. 股票是否同時滿足三項買入條件

### Q: 買入後立即賣出？
**A:** 可能原因：
1. 進場後價格跌超 -2% 觸發止損
2. 卻好進場時已達停利 +4%
3. 進場時間恰好是 13:25，觸發強制平倉

### Q: 如何修改停損/停利？
**A:** 編輯 `daytrade-bot.py` 中的 CONFIG：
```python
'STOP_LOSS': -0.02,      # 改成 -0.03 = -3%
'TAKE_PROFIT': 0.04,     # 改成 0.05 = +5%
```
然後重新啟動 bot

### Q: 能否同時持有多隻股票？
**A:** 可以。Bot 可以同時持有多個持倉，各自獨立管理

---

## 🎯 SMC 結構分析模組 (v3.0.0 新增)

### 概述
整合了 **Supply and Demand (SMC)** 市場結構分析引擎，提供高精度的進場和止盈點判斷。

### 核心功能

#### 1️⃣ 結構偵測
- **BOS (Break of Structure)**: 趨勢延續訊號
- **CHoCH (Change of Character)**: 趨勢反轉訊號
- **Sweep**: 掃止損訊號（影線刺穿、收盤回升）

#### 2️⃣ 失衡識別
- **FVG (Fair Value Gap)**: 三根K棒影線不重疊形成的失衡區
- **Mitigation 檢查**: 自動追蹤 FVG 是否已被回測
- **Order Block**: BOS/CHoCH 前的最後反方向 K 棒

#### 3️⃣ POI 三位一體
高概率入場區域需要同時滿足：
- 位移 (Displacement): 大實體 K 棒 (>1.5x 平均)
- 失衡 (Imbalance): 附近有未回測 FVG
- 流動性掃除 (Liquidity Sweep): 附近有 Sweep 訊號

#### 4️⃣ 多時框評分系統
整合 1H / 15M / 5M 多時框分析，給出 0-100 的 SMC 評分

#### 5️⃣ 階梯止盈系統 (LadderTP)
- TP1 @ 1.5×ATR: 平倉 50%，停損移保本
- TP2 @ 波段目標: 平倉 30%
- TP3 @ 昨高/昨低: 平倉剩餘 20%
- 移動停利追漲

#### 6️⃣ 日損控制 (DailyRiskManager)
- 日損上限：帳戶 3%
- 單次風險上限：帳戶 1%
- 連虧停損：連續 3 筆後冷靜
- 自動部位計算

#### 7️⃣ Kelly Criterion 資金管理
計算最佳風險比例，推薦使用 Half Kelly

### 測試結果 (2026-02-15)
✅ 所有 10 項測試通過 (100%)
- 波段高低點辨識 ✓
- 結構偵測 (BOS/CHoCH/Sweep) ✓
- FVG 失衡偵測 ✓
- Order Block 識別 ✓
- POI 三位一體 ✓
- SMC 綜合評分 ✓
- 階梯止盈系統 ✓
- 日損控制管理 ✓
- Kelly Criterion ✓
- 代碼安全檢查 ✓

### 模擬交易模式
✅ **本模組所有交易操作均為模擬交易**
🚫 **禁止使用真實資金或 Shioaji API 下單**

---

## 📞 支持

有問題或建議，請聯繫開發者或查看日誌檔案排查問題。

---

**最後更新：** 2026-02-15  
**版本：** 3.0.0
