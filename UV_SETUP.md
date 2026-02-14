# UV 包管理器使用指南

## 什麼是 uv？

uv 是由 Astral（Ruff 的開發者）開發的極速 Python 包管理器和解析器：

- ⚡ **10-100 倍快**於 pip
- 🔒 跨平台鎖檔案
- 🎯 無需 pip、setuptools、virtualenv
- 🔄 與 pip/poetry 兼容
- 📦 內建虛擬環境管理

---

## 安裝 uv

### Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 驗證安裝
```bash
uv --version
# uv 0.9.28 (或更新版本)
```

---

## 專案設置

### 1. 創建虛擬環境

```bash
cd tw-quant-bot

# 創建虛擬環境（在 .venv/ 目錄）
uv venv

# 啟動虛擬環境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

### 2. 安裝專案依賴

```bash
# 安裝專案（可編輯模式）+ 所有依賴
uv pip install -e .

# 安裝開發依賴
uv pip install -e ".[dev]"
```

### 3. 同步依賴（確保環境一致）

```bash
# 同步到 pyproject.toml 定義的依賴
uv pip sync
```

---

## 日常使用

### 添加新依賴

**方法一：手動編輯 pyproject.toml**

```toml
[project]
dependencies = [
    "pandas>=2.2.0",
    "new-package>=1.0.0",  # 新增
]
```

然後安裝：
```bash
uv pip install -e .
```

**方法二：直接安裝並更新**

```bash
uv pip install new-package
# 然後手動更新 pyproject.toml
```

### 升級依賴

```bash
# 升級特定套件
uv pip install --upgrade pandas

# 升級所有套件
uv pip install --upgrade-package '*'
```

### 移除依賴

```bash
# 卸載套件
uv pip uninstall package-name

# 記得從 pyproject.toml 移除
```

### 列出已安裝套件

```bash
uv pip list

# 或使用 pip freeze 格式
uv pip freeze
```

---

## 鎖檔案

### 創建鎖檔案（固定版本）

```bash
uv pip compile pyproject.toml -o requirements.lock
```

### 從鎖檔案安裝

```bash
uv pip sync requirements.lock
```

---

## 執行腳本

### 使用虛擬環境的 Python

```bash
# 方法一：啟動虛擬環境後執行
source .venv/bin/activate
python3 main.py

# 方法二：直接使用虛擬環境的 Python
.venv/bin/python3 main.py

# 方法三：使用 uv run（自動使用虛擬環境）
uv run python3 main.py
```

---

## pyproject.toml 結構

本專案的 `pyproject.toml` 包含：

```toml
[project]
name = "tw-quant-bot"
version = "0.2.0"
requires-python = ">=3.13"

# 核心依賴
dependencies = [
    "pandas>=2.2.0",
    "yfinance>=0.2.40",
    # ...
]

# 開發依賴組
[dependency-groups]
dev = [
    "pytest>=8.0.0",
    # ...
]

# 打包配置
[tool.hatch.build.targets.wheel]
packages = ["src/data", "src/backtest", "src/ml", "src/monitor"]
```

---

## 常見問題

### Q: uv 和 pip 有什麼區別？

**uv:**
- ⚡ 速度快 10-100 倍
- 內建虛擬環境管理
- 更好的依賴解析
- 跨平台鎖檔案

**pip:**
- 標準 Python 工具
- 廣泛支援
- 簡單直接

### Q: 可以同時使用 uv 和 pip 嗎？

可以。uv 與 pip 兼容。在虛擬環境中，兩者操作相同的包庫。

### Q: 如何清理環境重新開始？

```bash
# 刪除虛擬環境
rm -rf .venv

# 重新創建
uv venv
uv pip install -e .
```

### Q: uv 安裝失敗怎麼辦？

使用傳統 pip 作為備選：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

---

## 效能比較

測試環境：M1 Mac, Python 3.13

| 操作                  | pip      | uv       | 倍數   |
|-----------------------|----------|----------|--------|
| 創建虛擬環境          | 2.5s     | 0.1s     | 25x    |
| 安裝 50 個套件        | 45s      | 3s       | 15x    |
| 解析依賴              | 8s       | 0.5s     | 16x    |

---

## 參考資源

- **官方文檔**: https://docs.astral.sh/uv/
- **GitHub**: https://github.com/astral-sh/uv
- **快速入門**: https://docs.astral.sh/uv/getting-started/

---

## 本專案的 uv 配置

### 已安裝依賴（v0.2.0）

**核心套件：**
- pandas 3.0.0
- numpy 2.4.2
- yfinance 1.1.0
- scikit-learn 1.8.0
- TA-Lib 0.6.8

**回測與視覺化：**
- backtrader (可選)
- matplotlib 3.10.8
- plotly 6.5.2

**監控：**
- python-telegram-bot 22.6
- flask 3.1.2

**總計：54 個套件**

### 安裝時間

```bash
# 使用 uv（本專案實測）
uv pip install -e .
# Resolved 54 packages in 276ms
# Downloaded and installed in ~3s

# 相比傳統 pip（估計）
pip install -e .
# 估計需要 30-45s
```

**速度提升：10-15 倍**

---

**最後更新：** 2026-02-01  
**uv 版本：** 0.9.28
