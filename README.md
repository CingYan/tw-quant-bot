# tw-quant-bot — 台美股當沖交易系統

**版本：** v2.0.1  
**狀態：** ✅ 運行中（2026-03-18 驗證完成）  
**開發者：** Eagle + 羽燕鋒

---

## 🎯 專案簡介

自動化當沖交易系統，支援台股（Shioaji）與美股（Alpaca Paper Trading）。

### 核心架構（v2.0 重寫）

- 🧠 **SMC Engine** — Smart Money Concept 分析（OB/FVG/BOS/CHoCH）
- 📊 **VWAP Calculator** — 日內累積式 VWAP + σ 偏差帶 + Bounce 偵測
- 🛡️ **Risk Manager** — Kelly 部位計算、日損上限、連虧停機
- 📱 **Telegram Notifier** — 每筆進出場即時通知
- 📝 **Trade Recorder** — 進場即刻寫入磁碟（不等收盤）
- 🔒 **Failsafe** — 獨立強制平倉腳本，確保不帶倉過夜

---

## 📦 版本歷程

遵循 [語意化版本控制 (SemVer)](https://semver.org/lang/zh-TW/)

| 版本 | 日期 | 類型 | 說明 |
|------|------|------|------|
| 2.0.1 | 2026-03-18 | PATCH | trade_id KeyError 修復、DELETE 404 fallback、monitor crash protection |
| 2.0.0 | 2026-03-17 | MAJOR | 全面重寫：共用核心模組、watchdog daemon、failsafe、連虧停機 |
| 1.0.0 | 2026-03-12 | MAJOR | 初始版本：Shioaji native trading + SMC + VWAP |

---

## 📂 目錄結構

```
tw-quant-bot/
├── bots/
│   ├── tw-daytrade-bot.py       # 台股當沖 Bot（Shioaji）
│   ├── us-daytrade-bot.py       # 美股當沖 Bot（Alpaca）
│   ├── failsafe-close-tw.py     # 台股獨立強制平倉
│   ├── failsafe-close-us.py     # 美股獨立強制平倉
│   └── v2/core/                 # 共用核心模組
│       ├── smc_engine.py        # SMC 分析引擎
│       ├── risk_manager.py      # 風控管理
│       ├── notification.py      # Telegram 通知
│       └── trade_recorder.py    # 交易記錄
├── credentials/
│   ├── shioaji-credentials.example.json
│   └── alpaca-credentials.example.json
└── README.md
```

---

## 🚀 使用方式

### 前置需求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 套件管理
- Shioaji 帳號（台股）或 Alpaca Paper Trading 帳號（美股）

### 執行

```bash
# 台股 Bot
uv run python3 bots/tw-daytrade-bot.py run         # 主程式（daemon 模式）
uv run python3 bots/tw-daytrade-bot.py screener     # 盤前篩選
uv run python3 bots/tw-daytrade-bot.py log          # 查看今日交易
uv run python3 bots/tw-daytrade-bot.py status       # 查看持倉

# 美股 Bot
uv run python3 bots/us-daytrade-bot.py run
uv run python3 bots/us-daytrade-bot.py screener
uv run python3 bots/us-daytrade-bot.py log
uv run python3 bots/us-daytrade-bot.py status

# 獨立強制平倉（由 cron 執行，不依賴主 Bot）
uv run python3 bots/failsafe-close-tw.py
uv run python3 bots/failsafe-close-us.py
```

### 設定

複製 `credentials/*.example.json` 並填入你的 API 金鑰：

```bash
cp credentials/shioaji-credentials.example.json credentials/credentials.json
# 編輯 credentials.json，填入 api_key 和 secret_key
```

---

## 🛡️ 風控機制

| 機制 | 台股 | 美股 |
|------|------|------|
| 日損上限 | 資本 1.5% | 資本 5% / $10K |
| 最大持倉 | 3 檔 | 3 檔 |
| Kelly 比例 | 0.3 | 0.3 |
| 單筆風險上限 | 資本 0.5% | 資本 0.5% |
| 連虧停機 | 3 筆 → 當日停止 | 3 筆 → 當日停止 |
| 同標停損禁入 | ✅ 當日不再進場 | ✅ 當日不再進場 |
| 多時框衝突 | 硬拒絕 | 硬拒絕 |
| FVG 飽和 ≥7 | 硬拒絕 | 硬拒絕 |
| 強制平倉 | 13:20 | 15:50 ET |
| Failsafe | 獨立腳本 | 獨立腳本 |

---

## 📊 交易時段

### 台股
- **KZ1 開盤：** 09:00–10:00
- **KZ2 尾盤：** 12:30–13:20
- **強制平倉：** 13:20
- **禁止開倉：** 13:00 之後

### 美股
- **KZ1 開盤：** 09:30–10:30 ET
- **KZ2 午盤：** 12:00–13:30 ET
- **KZ3 收盤：** 14:30–15:50 ET
- **強制平倉：** 15:50 ET
- **禁止開倉：** 15:30 ET 之後

---

## ⚠️ 免責聲明

- 台股使用 Shioaji **模擬交易模式** (`simulation=True`)
- 美股使用 Alpaca **Paper Trading**
- 本系統僅供學習研究，不構成投資建議
- 使用者需自行承擔投資風險

---

## 📝 授權

MIT License

**最後更新：** 2026-03-18
