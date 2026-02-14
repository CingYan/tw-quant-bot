#!/usr/bin/env python3
"""
策略基類
"""

from typing import Dict, List, Optional
from .order import Order, OrderType, OrderSide, Portfolio


class Strategy:
    """策略基類"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.data: Dict[str, List[Dict]] = {}  # symbol -> 歷史資料
        self.current_index = 0
        self.signals: List[Dict] = []  # 訊號記錄
    
    def load_data(self, symbol: str, data: List[Dict]):
        """載入股票資料"""
        self.data[symbol] = data
    
    def get_current_bar(self, symbol: str) -> Optional[Dict]:
        """獲取當前 K 線資料"""
        if symbol not in self.data:
            return None
        
        if self.current_index >= len(self.data[symbol]):
            return None
        
        return self.data[symbol][self.current_index]
    
    def get_historical_bars(
        self, 
        symbol: str, 
        lookback: int = 20
    ) -> List[Dict]:
        """獲取歷史 K 線資料"""
        if symbol not in self.data:
            return []
        
        start = max(0, self.current_index - lookback + 1)
        end = self.current_index + 1
        
        return self.data[symbol][start:end]
    
    def buy(
        self, 
        symbol: str, 
        quantity: int, 
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> Order:
        """買入"""
        bar = self.get_current_bar(symbol)
        
        if not bar:
            raise ValueError(f"無法獲取 {symbol} 當前資料")
        
        # 市價單使用收盤價
        if order_type == OrderType.MARKET:
            price = bar['close']
        
        order = Order(
            symbol=symbol,
            order_type=order_type,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price
        )
        
        order.filled_price = price
        return order
    
    def sell(
        self, 
        symbol: str, 
        quantity: int, 
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> Order:
        """賣出"""
        bar = self.get_current_bar(symbol)
        
        if not bar:
            raise ValueError(f"無法獲取 {symbol} 當前資料")
        
        # 市價單使用收盤價
        if order_type == OrderType.MARKET:
            price = bar['close']
        
        order = Order(
            symbol=symbol,
            order_type=order_type,
            side=OrderSide.SELL,
            quantity=quantity,
            price=price
        )
        
        order.filled_price = price
        return order
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """平倉"""
        position = self.portfolio.get_position(symbol)
        
        if position.quantity == 0:
            return None
        
        return self.sell(symbol, position.quantity)
    
    def calculate_position_size(
        self, 
        symbol: str, 
        percentage: float = 0.2,
        allow_fractional: bool = True
    ) -> int:
        """
        計算倉位大小（基於總資產百分比）
        
        Args:
            symbol: 股票代碼
            percentage: 資金分配百分比（0.0-1.0）
            allow_fractional: 是否允許零股（< 1000 股）
        
        Returns:
            股數
        """
        bar = self.get_current_bar(symbol)
        
        if not bar:
            return 0
        
        # 獲取當前總資產
        prices = {s: self.get_current_bar(s)['close'] 
                 for s in self.data.keys() 
                 if self.get_current_bar(s)}
        
        total_value = self.portfolio.get_total_value(prices)
        
        # 計算可用於此倉位的資金
        position_value = total_value * percentage
        
        # 計算股數
        price = bar['close']
        shares = int(position_value / price)
        
        if shares == 0:
            return 0
        
        if allow_fractional:
            # 允許零股：直接返回股數
            return shares
        else:
            # 僅整張：轉換為張數（1張=1000股）
            lots = shares // 1000
            return lots * 1000 if lots > 0 else 0
    
    def record_signal(
        self, 
        symbol: str, 
        signal_type: str, 
        reason: str,
        metadata: Optional[Dict] = None
    ):
        """記錄交易訊號"""
        bar = self.get_current_bar(symbol)
        
        signal = {
            'date': bar['date'] if bar else None,
            'symbol': symbol,
            'type': signal_type,
            'reason': reason,
            'price': bar['close'] if bar else None,
            'metadata': metadata or {}
        }
        
        self.signals.append(signal)
    
    def on_bar(self):
        """
        每根 K 線回調（子類需要實作）
        
        在這裡實作交易邏輯：
        1. 分析當前市場狀況
        2. 生成買賣訊號
        3. 執行訂單
        """
        raise NotImplementedError("子類必須實作 on_bar 方法")
    
    def on_start(self):
        """回測開始時回調（可選實作）"""
        pass
    
    def on_end(self):
        """回測結束時回調（可選實作）"""
        pass


class MAStrategy(Strategy):
    """移動平均線交叉策略（範例）"""
    
    def __init__(
        self, 
        portfolio: Portfolio,
        short_period: int = 5,
        long_period: int = 20,
        position_pct: float = 0.2
    ):
        super().__init__(portfolio)
        self.short_period = short_period
        self.long_period = long_period
        self.position_pct = position_pct
    
    def calculate_ma(self, symbol: str, period: int) -> Optional[float]:
        """計算移動平均線"""
        bars = self.get_historical_bars(symbol, period)
        
        if len(bars) < period:
            return None
        
        prices = [bar['close'] for bar in bars]
        return sum(prices) / len(prices)
    
    def on_bar(self):
        """移動平均線交叉策略邏輯"""
        for symbol in self.data.keys():
            bar = self.get_current_bar(symbol)
            
            if not bar:
                continue
            
            # 使用預先計算的 MA（來自資料處理模組）
            ma_short = bar.get('ma_5')
            ma_long = bar.get('ma_20')
            
            if ma_short is None or ma_long is None:
                continue
            
            # 獲取前一根 K 線的 MA
            if self.current_index > 0:
                prev_bar = self.data[symbol][self.current_index - 1]
                prev_ma_short = prev_bar.get('ma_5')
                prev_ma_long = prev_bar.get('ma_20')
            else:
                prev_ma_short = None
                prev_ma_long = None
            
            position = self.portfolio.get_position(symbol)
            
            # 黃金交叉檢測：前一天 MA5 <= MA20，今天 MA5 > MA20
            if (prev_ma_short is not None and prev_ma_long is not None and
                prev_ma_short <= prev_ma_long and ma_short > ma_long and 
                position.quantity == 0):
                
                quantity = self.calculate_position_size(symbol, self.position_pct)
                
                if quantity > 0:
                    order = self.buy(symbol, quantity)
                    self.portfolio.execute_order(order)
                    
                    self.record_signal(
                        symbol, 
                        'BUY', 
                        f'黃金交叉 (MA{self.short_period}={ma_short:.2f} > MA{self.long_period}={ma_long:.2f})',
                        {'ma_short': ma_short, 'ma_long': ma_long}
                    )
            
            # 死亡交叉檢測：前一天 MA5 >= MA20，今天 MA5 < MA20
            elif (prev_ma_short is not None and prev_ma_long is not None and
                  prev_ma_short >= prev_ma_long and ma_short < ma_long and 
                  position.quantity > 0):
                
                order = self.close_position(symbol)
                
                if order:
                    self.portfolio.execute_order(order)
                    
                    self.record_signal(
                        symbol, 
                        'SELL', 
                        f'死亡交叉 (MA{self.short_period}={ma_short:.2f} < MA{self.long_period}={ma_long:.2f})',
                        {'ma_short': ma_short, 'ma_long': ma_long}
                    )


class RSIStrategy(Strategy):
    """RSI 超買超賣策略（範例）"""
    
    def __init__(
        self, 
        portfolio: Portfolio,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        position_pct: float = 0.2
    ):
        super().__init__(portfolio)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.position_pct = position_pct
    
    def on_bar(self):
        """RSI 策略邏輯"""
        for symbol in self.data.keys():
            bar = self.get_current_bar(symbol)
            
            if not bar:
                continue
            
            # 使用預先計算的 RSI（來自資料處理模組）
            rsi = bar.get('rsi')
            
            if rsi is None:
                continue
            
            position = self.portfolio.get_position(symbol)
            
            # RSI < 30：超賣 → 買入訊號
            if rsi < self.oversold and position.quantity == 0:
                quantity = self.calculate_position_size(symbol, self.position_pct)
                
                if quantity > 0:
                    order = self.buy(symbol, quantity)
                    self.portfolio.execute_order(order)
                    
                    self.record_signal(
                        symbol, 
                        'BUY', 
                        f'RSI 超賣 (RSI={rsi:.2f} < {self.oversold})',
                        {'rsi': rsi}
                    )
            
            # RSI > 70：超買 → 賣出訊號
            elif rsi > self.overbought and position.quantity > 0:
                order = self.close_position(symbol)
                
                if order:
                    self.portfolio.execute_order(order)
                    
                    self.record_signal(
                        symbol, 
                        'SELL', 
                        f'RSI 超買 (RSI={rsi:.2f} > {self.overbought})',
                        {'rsi': rsi}
                    )
