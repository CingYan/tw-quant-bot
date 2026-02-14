#!/bin/bash
# 假設性交易系統啟動腳本
# 九點開始運作，每 30 分鐘檢查一次訊號

echo "🚀 準備啟動假設性交易系統"
echo "========================================="
echo "⏰ 開始時間：09:00"
echo "💰 初始資金：100 萬 TWD"
echo "📊 交易標的：台積電 2330.TW"
echo "📱 訊息推送：柯姊敗家團"
echo "========================================="
echo ""

# 檢查當前時間
CURRENT_HOUR=$(date +%H)
CURRENT_MIN=$(date +%M)

if [ "$CURRENT_HOUR" -lt 9 ]; then
    WAIT_MIN=$(( (9 - CURRENT_HOUR) * 60 - CURRENT_MIN ))
    echo "⏰ 尚未到九點，等待 $WAIT_MIN 分鐘..."
    echo "💡 按 Ctrl+C 取消等待並立即開始"
    echo ""
    
    # 倒數計時
    for i in $(seq $WAIT_MIN -1 1); do
        echo -ne "⏳ 距離開始還有 $i 分鐘...\r"
        sleep 60
    done
    echo ""
fi

echo "✅ 九點到了！啟動交易系統..."
echo ""

# 啟動自動交易系統（每 30 分鐘檢查一次）
cd /home/node/clawd/tw-quant-bot
.venv/bin/python3 auto_trading.py --interval 30
