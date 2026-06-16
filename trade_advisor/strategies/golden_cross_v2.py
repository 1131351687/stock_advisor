"""
多金叉共振策略 V2

识别均线金叉、KDJ金叉、MACD金叉三者同时发生或相隔不到3天的共振信号

选股条件：
1. 均线金叉：短期均线上穿长期均线（如5日上穿20日）
2. KDJ金叉：K线上穿D线
3. MACD金叉：DIF线上穿DEA线
4. 共振确认：三个金叉信号同时发生或相隔不到3天
5. 回溯范围：在最近10天内寻找金叉信号
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# 直接导入基类和数据类（不依赖 strategies 模块以避免循环导入）
@dataclass
class StrategyParam:
    """策略参数定义"""
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
    """个股信号"""
    code: str
    name: str
    signal: str
    reason: str
    market_cap: float = 0
    weight: float = 0
    price: float = 0


@dataclass
class StrategyResult:
    """策略运行结果"""
    date: str
    strategy_name: str
    signals: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """策略基类"""

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


# 导入工具函数（从 strategies 模块）
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
        except Exception:
            continue
    last = cal[-1]
    ds = _to_date_str(last)
    return last, ds


def get_stock_display_name(stock):
    try:
        return stock.name.encode("latin-1").decode("gbk", errors="replace")
    except Exception:
        return stock.name


def _make_sell_signals(holdings_set, buy_codes, day_query):
    from hikyuu.interactive import sm, Query
    from hikyuu.interactive import Days
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
# 导入数据适配器（必须在 hikyuu.interactive 之后导入，避免覆盖）
from trade_advisor.data_adapter import get_kdata, get_stock_list, get_block_list


class MultiGoldenCrossStrategyV2(BaseStrategy):
    """多金叉共振策略 V2，使用 Hikyuu 数据源"""

    @property
    def name(self) -> str:
        return "多金叉共振 V2"

    @property
    def description(self) -> str:
        return "识别均线金叉、KDJ金叉、MACD金叉三者共振，降低误信号"

    @property
    def params(self) -> list:
        return [
            StrategyParam("ma_short", "短期均线周期", 5, "int", min_val=3, max_val=15),
            StrategyParam("ma_long", "长期均线周期", 20, "int", min_val=10, max_val=60),
            StrategyParam("kdj_n", "KDJ参数N", 9, "int", min_val=5, max_val=20),
            StrategyParam("macd_short", "MACD短EMA", 12, "int", min_val=6, max_val=20),
            StrategyParam("macd_long", "MACD长EMA", 26, "int", min_val=15, max_val=50),
            StrategyParam("lookback_days", "回溯天数", 10, "int", min_val=5, max_val=30),
            StrategyParam("max_stocks", "最多选股数", 30, "int", min_val=5, max_val=100),
        ]

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        """
        执行多金叉共振策略

        Args:
            param_values: 策略参数
            date: 选股日期，None=最新交易日
            holdings: 当前持仓（用于生成卖出信号）

        Returns:
            StrategyResult
        """
        # 参数提取
        ma_short = int(param_values.get("ma_short", 5))
        ma_long = int(param_values.get("ma_long", 20))
        kdj_n = int(param_values.get("kdj_n", 9))
        macd_short = int(param_values.get("macd_short", 12))
        macd_long = int(param_values.get("macd_long", 26))
        lookback_days = int(param_values.get("lookback_days", 10))
        max_stocks = int(param_values.get("max_stocks", 30))

        # 确定目标日期
        if date is not None:
            target_dt = _to_hikyuu_date(date)
        else:
            _, date_str = _get_last_trade_date()
            if date_str == "N/A":
                return StrategyResult(date="N/A", strategy_name=self.name,
                                     metadata={"error": "无法获取交易日期"})
            target_dt = _to_hikyuu_date(date_str)

        date_str = _to_date_str(target_dt)

        # 获取股票池 - 使用全市场 A 股
        try:
            stock_codes = get_stock_list('all')
            if not stock_codes:
                return StrategyResult(date=date_str, strategy_name=self.name,
                                     metadata={"error": "无法获取股票列表"})
        except Exception as e:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                 metadata={"error": f"获取股票列表失败: {e}"})

        # 扫描股票池，找出满足条件的股票
        buy_signals = []

        for code in stock_codes[:500]:  # 限制扫描数量，提高效率
            try:
                # 获取 K 线数据
                start_dt = (target_dt - Days(lookback_days + 50)).datetime().date()
                end_date = target_dt.datetime().date()

                df = get_kdata(code, str(start_dt), str(end_date))
                if df.empty or len(df) < lookback_days:
                    continue

                # 计算指标
                df = self._calculate_indicators(
                    df, ma_short, ma_long, kdj_n, macd_short, macd_long
                )

                # 选股逻辑
                signal_info = self._check_resonance(
                    df, code, lookback_days, target_dt
                )

                if signal_info:
                    buy_signals.append(signal_info)

                if len(buy_signals) >= max_stocks:
                    break

            except Exception as e:
                # 单只股票处理失败不影响其他股票
                continue

        # 排序：按共振强度排序
        buy_signals = sorted(
            buy_signals,
            key=lambda x: x.get('resonance_strength', 0),
            reverse=True
        )[:max_stocks]

        # 生成信号
        buy_codes = [s['code'] for s in buy_signals]
        signals = [
            StockSignal(
                code=s['code'],
                name=s['name'],
                signal='buy',
                reason=s['reason'],
                price=s['price'],
                weight=1.0 / len(buy_signals) if buy_signals else 0
            )
            for s in buy_signals
        ]

        # 添加卖出信号
        if holdings:
            holdings_set = set(holdings)
            sell_signals = _make_sell_signals(
                holdings_set, set(buy_codes),
                Query(target_dt, target_dt + Days(1))
            )
            signals.extend(sell_signals)

        return StrategyResult(
            date=date_str,
            strategy_name=self.name,
            signals=signals,
            metadata={
                "stock_count": len(buy_signals),
                "scan_total": min(len(stock_codes), 500)
            }
        )

    def _calculate_indicators(self, df, ma_short, ma_long, kdj_n, macd_short, macd_long):
        """计算技术指标"""
        df = df.sort_values('date', ascending=True).copy()

        close = df['close']
        high = df['high']
        low = df['low']

        # 均线
        df['ma_short'] = close.rolling(window=ma_short, min_periods=1).mean()
        df['ma_long'] = close.rolling(window=ma_long, min_periods=1).mean()

        # KDJ
        lowest_low = low.rolling(window=kdj_n, min_periods=1).min()
        highest_high = high.rolling(window=kdj_n, min_periods=1).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)

        k_values = rsv.ewm(alpha=1/3, adjust=False).mean()
        d_values = k_values.ewm(alpha=1/3, adjust=False).mean()

        df['K'] = k_values
        df['D'] = d_values

        # MACD
        ema_short = close.ewm(span=macd_short, adjust=False).mean()
        ema_long = close.ewm(span=macd_long, adjust=False).mean()
        df['DIF'] = ema_short - ema_long
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()

        # 金叉信号
        df['ma_cross'] = (df['ma_short'] > df['ma_long']) & \
                         (df['ma_short'].shift(1) <= df['ma_long'].shift(1))
        df['kdj_cross'] = (df['K'] > df['D']) & \
                          (df['K'].shift(1) <= df['D'].shift(1))
        df['macd_cross'] = (df['DIF'] > df['DEA']) & \
                           (df['DIF'].shift(1) <= df['DEA'].shift(1))

        return df.sort_values('date', ascending=False)

    def _check_resonance(self, df, code, lookback_days, target_dt):
        """检查多金叉共振条件"""
        if df.empty:
            return None

        latest = df.iloc[0]

        # 基本检查
        if latest['volume'] <= 0 or pd.isna(latest['close']):
            return None

        # 检查回溯期内是否有三个金叉
        lookback_df = df.head(lookback_days)

        ma_cross_idx = None
        kdj_cross_idx = None
        macd_cross_idx = None

        # 找三个金叉最近的出现位置
        for i, (idx, row) in enumerate(lookback_df.iterrows()):
            if row['ma_cross'] and ma_cross_idx is None:
                ma_cross_idx = i
            if row['kdj_cross'] and kdj_cross_idx is None:
                kdj_cross_idx = i
            if row['macd_cross'] and macd_cross_idx is None:
                macd_cross_idx = i

        # 三个金叉必须都存在
        if ma_cross_idx is None or kdj_cross_idx is None or macd_cross_idx is None:
            return None

        # 计算时间差
        max_diff = max(abs(ma_cross_idx - kdj_cross_idx),
                      abs(kdj_cross_idx - macd_cross_idx),
                      abs(ma_cross_idx - macd_cross_idx))

        # 共振强度：差值越小越强
        resonance_strength = 10 - max_diff

        if resonance_strength <= 0:
            return None

        # 获取股票名称
        try:
            stock = sm[code.lower()]
            name = stock.name.encode("latin-1").decode("gbk", errors="replace")
        except:
            name = code

        price = float(latest['close'])

        return {
            'code': code,
            'name': name,
            'price': price,
            'reason': f"多金叉共振 (强度{resonance_strength})",
            'resonance_strength': resonance_strength,
            'ma_cross': latest['ma_cross'],
            'kdj_cross': latest['kdj_cross'],
            'macd_cross': latest['macd_cross'],
        }
