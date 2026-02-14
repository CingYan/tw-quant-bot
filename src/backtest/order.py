#!/usr/bin/env python3
"""
訂單與持倉管理模組
"""

from enum import Enum
from datetime import datetime
from typing import Optional


class OrderType(Enum):
    """訂單類型"""
    MARKET = "market"       # 市價單
    LIMIT = "limit"         # 限價單


class OrderSide(Enum):
    """買賣方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """訂單狀態"""
    CREATED = "created"     # 已創建
    SUBMITTED = "submitted" # 已提交
    FILLED = "filled"       # 已成交
    CANCELLED = "cancelled" # 已取消
    REJECTED = "rejected"   # 已拒絕


class Order:
    """訂單"""
    
    def __init__(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: int,
        price: Optional[float] = None,
        created_at: Optional[datetime] = None
    ):
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price  # 限價單才需要
        self.created_at = created_at or datetime.now()
        self.status = OrderStatus.CREATED
        self.filled_price: Optional[float] = None
        self.filled_at: Optional[datetime] = None
        self.commission = 0.0
        self.tax = 0.0
    
    def fill(self, price: float, commission: float = 0.0, tax: float = 0.0):
        """成交"""
        self.filled_price = price
        self.filled_at = datetime.now()
        self.commission = commission
        self.tax = tax
        self.status = OrderStatus.FILLED
    
    def cancel(self):
        """取消"""
        self.status = OrderStatus.CANCELLED
    
    def reject(self):
        """拒絕"""
        self.status = OrderStatus.REJECTED
    
    @property
    def total_cost(self) -> float:
        """總成本（含手續費和稅金）"""
        if not self.filled_price:
            return 0.0
        
        base_cost = self.filled_price * self.quantity
        return base_cost + self.commission + self.tax
    
    def __repr__(self):
        return (
            f"Order({self.symbol} {self.side.value} "
            f"{self.quantity}@{self.price or 'MARKET'} "
            f"status={self.status.value})"
        )


class Position:
    """持倉"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = 0
        self.avg_price = 0.0
        self.total_cost = 0.0
        self.realized_pnl = 0.0  # 已實現損益
    
    def add(self, quantity: int, price: float, commission: float = 0.0):
        """加倉"""
        cost = price * quantity + commission
        
        if self.quantity == 0:
            self.avg_price = price
            self.total_cost = cost
        else:
            # 更新平均成本
            total_value = self.total_cost + cost
            self.quantity += quantity
            self.avg_price = (total_value - commission) / self.quantity
            self.total_cost = total_value
        
        self.quantity += quantity
    
    def reduce(
        self, 
        quantity: int, 
        price: float, 
        commission: float = 0.0,
        tax: float = 0.0
    ) -> float:
        """減倉（返回已實現損益）"""
        if quantity > self.quantity:
            raise ValueError(f"減倉數量 {quantity} 超過持倉 {self.quantity}")
        
        # 計算已實現損益
        revenue = price * quantity - commission - tax
        cost = self.avg_price * quantity
        pnl = revenue - cost
        
        self.realized_pnl += pnl
        self.quantity -= quantity
        
        if self.quantity == 0:
            self.avg_price = 0.0
            self.total_cost = 0.0
        else:
            self.total_cost -= cost
        
        return pnl
    
    def unrealized_pnl(self, current_price: float) -> float:
        """未實現損益"""
        if self.quantity == 0:
            return 0.0
        
        current_value = current_price * self.quantity
        return current_value - self.total_cost
    
    def __repr__(self):
        return (
            f"Position({self.symbol} {self.quantity}@{self.avg_price:.2f} "
            f"cost={self.total_cost:.2f} realized_pnl={self.realized_pnl:.2f})"
        )


class Portfolio:
    """投資組合"""
    
    def __init__(self, initial_cash: float = 1000000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.total_commission = 0.0
        self.total_tax = 0.0
    
    def get_position(self, symbol: str) -> Position:
        """獲取持倉（不存在則創建）"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]
    
    def execute_order(
        self,
        order: Order,
        commission_rate: float = 0.001425,
        tax_rate: float = 0.003
    ):
        """執行訂單"""
        if order.status != OrderStatus.CREATED:
            return
        
        # 計算手續費（買賣都收）
        commission = order.filled_price * order.quantity * commission_rate
        
        # 計算證交稅（僅賣出收）
        tax = 0.0
        if order.side == OrderSide.SELL:
            tax = order.filled_price * order.quantity * tax_rate
        
        # 檢查資金是否足夠（買入時）
        if order.side == OrderSide.BUY:
            total_cost = order.filled_price * order.quantity + commission
            if total_cost > self.cash:
                order.reject()
                return
            
            self.cash -= total_cost
            position = self.get_position(order.symbol)
            position.add(order.quantity, order.filled_price, commission)
        
        # 賣出
        else:
            position = self.get_position(order.symbol)
            
            # 檢查持倉是否足夠
            if order.quantity > position.quantity:
                order.reject()
                return
            
            revenue = order.filled_price * order.quantity - commission - tax
            self.cash += revenue
            position.reduce(order.quantity, order.filled_price, commission, tax)
        
        # 成交訂單
        order.fill(order.filled_price, commission, tax)
        self.orders.append(order)
        
        self.total_commission += commission
        self.total_tax += tax
    
    def get_total_value(self, prices: dict[str, float]) -> float:
        """獲取總資產價值"""
        total = self.cash
        
        for symbol, position in self.positions.items():
            if position.quantity > 0 and symbol in prices:
                total += prices[symbol] * position.quantity
        
        return total
    
    def get_equity_curve(self, prices: dict[str, float]) -> float:
        """獲取當前權益（現金 + 未實現損益）"""
        return self.get_total_value(prices)
    
    def __repr__(self):
        return (
            f"Portfolio(cash={self.cash:.2f} "
            f"positions={len([p for p in self.positions.values() if p.quantity > 0])} "
            f"commission={self.total_commission:.2f} tax={self.total_tax:.2f})"
        )
