# 快速入門指南

## 5 分鐘上手台股量化交易系統

### 步驟 0：安裝依賴（使用 uv）

```bash
# 進入專案目錄
cd /home/node/clawd/tw-quant-bot

# 創建虛擬環境
uv venv

# 啟動虛擬環境
source .venv/bin/activate

# 安裝所有依賴
uv pip install -e .
```

### 步驟 1：執行測試

確認環境正常：

```bash
cd /home/node/clawd/tw-quant-bot
python3 test_basic.py
```

預期輸出：
```
✅ 資料擷取: 通過
✅ 資料處理: 通過
🎉 所有測試通過！
```

### 步驟 2：更新股票資料

更新配置中的所有股票（預設：2330.TW, 2317.TW, 2454.TW, 2412.TW, 1301.TW）：

```bash
python3 main.py all
```

或更新特定股票：

```bash
python3 main.py update --symbols 2330.TW 2454.TW --days 180
```

### 步驟 3：查看資料

```bash
# 查看台積電最近 20 筆資料
python3 main.py show 2330.TW --limit 20
```

輸出範例：
```
日期         收盤     成交量       RSI      MA5      MA20     量比    
--------------------------------------------------------------------------------
2025-01-20   1125.00  37842598     61.54    1112.00  N/A      N/A     
2025-01-21   1130.00  41234567     63.21    1115.00  N/A      N/A     
2025-01-22   1135.00  45678901     65.22    1120.00  N/A      N/A     
```

### 步驟 4：檢查資料庫

```bash
sqlite3 data/tw_quant.db "SELECT symbol, COUNT(*) as count FROM stock_data GROUP BY symbol;"
```

### 步驟 5：查看處理後的資料

```bash
# JSON 格式
cat data/processed/2330.TW.json | head -50
```

---

## 常用指令

### 資料更新

```bash
# 更新所有股票（最近 365 天）
python3 main.py update

# 更新特定股票（最近 180 天）
python3 main.py update --symbols 2330.TW 2454.TW --days 180

# 一鍵更新並處理
python3 main.py all
```

### 資料處理

```bash
# 處理所有股票
python3 main.py process

# 處理特定股票
python3 main.py process --symbols 2330.TW
```

### 資料查詢

```bash
# 顯示最近 10 筆資料
python3 main.py show 2330.TW

# 顯示最近 50 筆資料
python3 main.py show 2330.TW --limit 50
```

### 資料庫操作

```bash
# 進入 SQLite 互動模式
sqlite3 data/tw_quant.db

# 常用 SQL 查詢
sqlite3 data/tw_quant.db "SELECT * FROM stock_data WHERE symbol='2330.TW' ORDER BY date DESC LIMIT 5;"
```

---

## 配置修改

編輯 `config/config.json`：

```json
{
  "data": {
    "symbols": [
      "2330.TW",    // 台積電
      "2317.TW",    // 鴻海
      "2454.TW"     // 聯發科
    ],
    "update_interval": 3600
  }
}
```

---

## 故障排除

### 問題 1：無法連線到 Yahoo Finance

**症狀：** HTTP 錯誤或 Timeout

**解決方法：**
1. 檢查網路連線
2. 確認股票代碼正確（台股必須加 `.TW`）
3. 稍後重試（可能遇到 rate limit）

### 問題 2：技術指標為 None

**症狀：** RSI、MA20、Volume Ratio 顯示 N/A

**原因：** 資料筆數不足

**解決方法：**
```bash
# 增加回溯天數
python3 main.py update --days 365
```

### 問題 3：資料庫鎖定

**症狀：** `database is locked`

**解決方法：**
```bash
# 確保沒有其他程式正在使用資料庫
lsof data/tw_quant.db
```

---

## 下一步

1. **執行完整資料更新**
   ```bash
   python3 main.py all
   ```

2. **檢視專案進度**
   ```bash
   cat PROGRESS.md
   ```

3. **開始實作階段二（Backtrader 回測引擎）**
   - 參考 README.md 的「階段二：回測引擎」

---

**需要協助？** 查看 `README.md` 或 `PROGRESS.md`
