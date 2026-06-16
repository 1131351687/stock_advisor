"""
涨停回调策略 - 迁移自 KHunter，已适配 Hikyuu 数据源

逻辑：
- 识别涨停的股票
- 在次日回调低于前期高点时买入
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


class LimitUpPullbackStrategy(BaseStrategy):
    """涨停回调策略 - 识别涨停回调的买点"""

    @property
    def name(self) -> str:
        return "涨停回调"

    @property
    def description(self) -> str:
        return "捕捉涨停后回调的低风险买点"

    @property
    def params(self) -> list:
        return [
            StrategyParam("lookback_days", "回溯天数", 20, "int", min_val=10, max_val=60),
            StrategyParam("pullback_ratio", "回调比例", 5.0, "float", min_val=1, max_val=15),
            StrategyParam("max_stocks", "最多选股数", 30, "int", min_val=5, max_val=100),
        ]

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        """执行涨停回调策略"""
        lookback_days = int(param_values.get("lookback_days", 20))
        pullback_ratio = float(param_values.get("pullback_ratio", 5.0))
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

        # 获取股票池 - 使用中小板（涨停回调常见）
        try:
            stock_codes = get_stock_list('small_cap')[:300]
        except:
            stock_codes = []

        buy_signals = []

        # 扫描股票池
        for code in stock_codes:
            try:
                start = (target_dt - Days(lookback_days + 10)).datetime().date()
                end = target_dt.datetime().date()
                df = get_kdata(code, str(start), str(end))
                if len(df) < 5:
                    continue

                signal = self._detect_limit_up_pullback(df, code, pullback_ratio)
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
                       reason=f"涨停回调 (回调{s['pullback']:.1f}%)",
                       price=s['price'],
                       weight=1.0 / len(buy_signals) if buy_signals else 0)
            for s in buy_signals
        ]

        if holdings:
            sell_signals = _make_sell_signals(set(holdings), set(buy_codes),
                                            Query(target_dt, target_dt + Days(1)))
            signals.extend(sell_signals)

        return StrategyResult(date=date_str, strategy_name=self.name, signals=signals)

    def _detect_limit_up_pullback(self, df, code, pullback_ratio):
        """检测涨停回调"""
        if len(df) < 2:
            return None

        df = df.sort_values('date', ascending=True)
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values

        today_close = closes[-1]
        today_open = opens[-1]
        today_low = lows[-1]

        # 检查是否是涨停日（收盘接近涨停）
        if len(closes) >= 2:
            prev_close = closes[-2]
            up_limit = prev_close * 1.095  # 涨停价

            # 今日是否有涨停或大涨
            if highs[-2] >= up_limit * 0.99:  # 前一天达到涨停
                # 今日回调
                pullback_pct = (highs[-2] - today_low) / highs[-2] * 100

                if pullback_pct >= pullback_ratio * 0.5:  # 回调幅度足够
                    if today_close > today_open:  # 今日收红
                        try:
                            stock = sm[code.lower()]
                            name = get_stock_display_name(stock)
                        except:
                            name = code

                        return {
                            'code': code,
                            'name': name,
                            'price': float(today_close),
                            'pullback': pullback_pct,
                        }

        return None
