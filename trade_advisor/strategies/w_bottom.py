"""
W 底策略 - 迁移自 KHunter，已适配 Hikyuu 数据源

W 底形态识别：
- 两个低点大致相等
- 中间有高点
- 突破右侧高点时产生买信号
"""

import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# 直接定义基类和数据类（避免循环导入）
@dataclass
class StrategyParam:
    name: str
    label: str
    default: Any
    param_type: str = "int"
    options: list = None
    min_val: float = None
    max_val: float = None
    step: float = None


@dataclass
class StockSignal:
    code: str
    name: str
    signal: str
    reason: str
    market_cap: float = 0
    weight: float = 0
    price: float = 0


@dataclass
class StrategyResult:
    date: str
    strategy_name: str
    signals: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def params(self) -> list: ...

    @abstractmethod
    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult: ...


# 导入工具函数
def _to_hikyuu_date(date_input):
    from hikyuu.interactive import Datetime, Days
    if isinstance(date_input, Datetime):
        return date_input
    if isinstance(date_input, str):
        parts = date_input.split("-")
        return Datetime(int(parts[0]), int(parts[1]), int(parts[2]))
    if hasattr(date_input, "year"):
        return Datetime(date_input.year, date_input.month, date_input.day)
    return Datetime.now()


def _to_date_str(date_input):
    dt = _to_hikyuu_date(date_input)
    return str(dt)[:10]


def _get_last_trade_date(lookback=30):
    from hikyuu.interactive import sm, Query, Days
    cal = sm.get_trading_calendar(Query(-lookback))
    if not cal or len(cal) == 0:
        return None, "N/A"
    ref_stock = sm["sh510050"]
    for i in range(min(len(cal), 10)):
        dt = cal[-(i + 1)]
        q = Query(dt, dt + Days(1))
        try:
            k = ref_stock.get_kdata(q)
            if k and len(k) > 0 and float(k[-1].close) > 0:
                ds = _to_date_str(dt)
                return dt, ds
        except:
            continue
    return cal[-1], _to_date_str(cal[-1])


def get_stock_display_name(stock):
    try:
        return stock.name.encode("latin-1").decode("gbk", errors="replace")
    except:
        return stock.name


def _make_sell_signals(holdings_set, buy_codes, day_query):
    from hikyuu.interactive import sm
    buy_set = set(buy_codes)
    signals = []
    for code in sorted(holdings_set - buy_set):
        try:
            hk_code = code[:2].lower() + code[2:]
            stk = sm[hk_code]
            name = get_stock_display_name(stk)
            k = stk.get_kdata(day_query)
            price = float(k[-1].close) if k and len(k) > 0 else 0
        except:
            name = code
            price = 0
        signals.append(StockSignal(
            code=code, name=name, signal="sell",
            reason="调出目标列表",
            market_cap=0, price=price, weight=0,
        ))
    return signals


from hikyuu.interactive import *
from trade_advisor.data_adapter import get_kdata, get_stock_list


class WBottomStrategy(BaseStrategy):
    """W 底策略 - 识别 W 底形态，提示买入机会"""

    @property
    def name(self) -> str:
        return "W 底形态"

    @property
    def description(self) -> str:
        return "识别股价 W 底形态，捕捉底部反转机会"

    @property
    def params(self) -> list:
        return [
            StrategyParam("lookback_days", "回溯天数", 30, "int", min_val=15, max_val=60),
            StrategyParam("deviation", "低点偏差百分比", 5.0, "float", min_val=1, max_val=10),
            StrategyParam("max_stocks", "最多选股数", 30, "int", min_val=5, max_val=100),
        ]

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        """执行 W 底策略"""
        lookback_days = int(param_values.get("lookback_days", 30))
        deviation = float(param_values.get("deviation", 5.0))
        max_stocks = int(param_values.get("max_stocks", 30))

        # 确定目标日期
        if date is not None:
            target_dt = _to_hikyuu_date(date)
        else:
            _, date_str = _get_last_trade_date()
            if date_str == "N/A":
                return StrategyResult(date="N/A", strategy_name=self.name)
            target_dt = _to_hikyuu_date(date_str)

        date_str = _to_date_str(target_dt)

        # 获取股票池
        try:
            stock_codes = get_stock_list('small_cap')[:300]
        except:
            stock_codes = []

        buy_signals = []

        # 扫描股票池
        for code in stock_codes:
            try:
                start = (target_dt - Days(lookback_days + 30)).datetime().date()
                end = target_dt.datetime().date()
                df = get_kdata(code, str(start), str(end))
                if len(df) < lookback_days:
                    continue

                signal = self._detect_w_bottom(df, code, deviation)
                if signal:
                    buy_signals.append(signal)

                if len(buy_signals) >= max_stocks:
                    break
            except:
                continue

        # 生成信号
        buy_codes = [s['code'] for s in buy_signals]
        signals = [
            StockSignal(code=s['code'], name=s['name'], signal='buy',
                       reason=f"W底形态 ({s['score']:.1f})", price=s['price'],
                       weight=1.0 / len(buy_signals) if buy_signals else 0)
            for s in buy_signals
        ]

        if holdings:
            sell_signals = _make_sell_signals(set(holdings), set(buy_codes),
                                            Query(target_dt, target_dt + Days(1)))
            signals.extend(sell_signals)

        return StrategyResult(date=date_str, strategy_name=self.name, signals=signals)

    def _detect_w_bottom(self, df, code, deviation):
        """检测 W 底形态"""
        if len(df) < 10:
            return None

        df = df.sort_values('date', ascending=True)
        lows = df['low'].values
        highs = df['high'].values
        closes = df['close'].values
        dates = df['date'].values

        # 简化的 W 底检测：找两个局部低点
        for i in range(2, len(lows) - 3):
            left_low = lows[i-2:i].min()
            mid_high = highs[i:i+2].max()
            right_low = lows[i+2:i+4].min()

            # 检查 W 底条件
            if abs(left_low - right_low) / left_low < deviation / 100:
                if mid_high > left_low * 1.02:  # 中间高点显著
                    if closes[-1] > mid_high * 0.98:  # 正在上升
                        try:
                            stock = sm[code.lower()]
                            name = get_stock_display_name(stock)
                        except:
                            name = code

                        return {
                            'code': code,
                            'name': name,
                            'price': float(closes[-1]),
                            'score': 8.0,
                        }

        return None
