"""
策略模块 — 各策略的实现

包括：
1. 原有策略（5个）
2. 迁移自 KHunter 的策略（基于 Hikyuu 数据源）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import numpy as np
from hikyuu.interactive import *


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
    signal: str           # buy / sell / hold
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


def _to_hikyuu_date(date_input):
    """将 str / datetime / Datetime 统一转为 Datetime"""
    if isinstance(date_input, Datetime):
        return date_input
    if isinstance(date_input, str):
        parts = date_input.split("-")
        return Datetime(int(parts[0]), int(parts[1]), int(parts[2]))
    if hasattr(date_input, "year"):
        return Datetime(date_input.year, date_input.month, date_input.day)
    return Datetime.now()


def _to_date_str(date_input):
    """统一转为 'YYYY-MM-DD' 字符串"""
    dt = _to_hikyuu_date(date_input)
    return str(dt)[:10]


def _get_last_trade_date(lookback=30):
    """
    获取最近一个确有 K 线数据的交易日。

    交易日历可能已更新但 K 线数据尚未下载，因此需要交叉验证。
    从日历的最后一个日期往前遍历，找到有实际 K 线数据的日期。
    """
    cal = sm.get_trading_calendar(Query(-lookback))
    if not cal or len(cal) == 0:
        return None, "N/A"

    # 用 510050 作为参考 ETF 来检查 KData 可用性
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
    # 退回到日历最后一个日期（兜底）
    last = cal[-1]
    ds = _to_date_str(last)
    return last, ds


def get_stock_display_name(stock):
    try:
        return stock.name.encode("latin-1").decode("gbk", errors="replace")
    except Exception:
        return stock.name


def is_st_stock(stock):
    name = get_stock_display_name(stock)
    return "ST" in name or "*" in name or "退" in name


def _lookup_stock_info(code, day_query):
    """通过股票代码查询名称和价格，用于卖出信号"""
    try:
        hk_code = code[:2].lower() + code[2:]
        stk = sm[hk_code]
        name = get_stock_display_name(stk)
        k = stk.get_kdata(day_query)
        price = float(k[-1].close) if k and len(k) > 0 else 0
        return name, price
    except Exception:
        return code, 0


def _make_sell_signals(holdings_set, buy_codes, day_query):
    """生成卖出信号（含名称和价格查询）"""
    buy_set = set(buy_codes)
    signals = []
    for code in sorted(holdings_set - buy_set):
        name, price = _lookup_stock_info(code, day_query)
        signals.append(StockSignal(
            code=code, name=name, signal="sell",
            reason="调出目标列表",
            market_cap=0, price=price, weight=0,
        ))
    return signals


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
    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        """date: None 表示最新交易日, 否则为指定日期
           holdings: 当前持仓代码列表, 用于生成卖出/持有信号"""
        ...


class SmallCapStrategy(BaseStrategy):
    """小市值策略 — 中小板指中选流通市值最小的N只"""

    @property
    def name(self):
        return "小市值策略"

    @property
    def description(self):
        return "从中小板指(399101)成分股中，选取流通市值最小的N只股票，过滤ST/涨跌停/停牌"

    @property
    def params(self):
        return [
            StrategyParam("buy_count", "持仓数量", 5, "int", min_val=1, max_val=20),
            StrategyParam("pool_size", "候选池倍数", 3, "int", min_val=1, max_val=10),
            StrategyParam("max_cap", "最大市值(亿)", 50, "float", min_val=1, max_val=1000),
        ]

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        buy_count = int(param_values.get("buy_count", 5))
        pool_mult = int(param_values.get("pool_size", 3))
        max_cap = float(param_values.get("max_cap", 50)) * 1e8

        # 确定目标日期
        if date is not None:
            target_dt = _to_hikyuu_date(date)
        else:
            _, date_str = _get_last_trade_date()
            if date_str == "N/A":
                return StrategyResult(date="N/A", strategy_name=self.name,
                                      metadata={"error": "无法获取交易日历"})
            target_dt = _to_hikyuu_date(date_str)
        date_str = _to_date_str(target_dt)

        # 检查是否为交易日
        cal = sm.get_trading_calendar(Query(target_dt, target_dt + Days(1)))
        if not cal or len(cal) == 0:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "当日非交易日或数据未更新"})

        # 获取当日数据（精确查询该日期）
        day_query = Query(target_dt, target_dt + Days(1))

        # 获取股票池
        idx_stock = sm["sz399101"]
        blocks = sm.get_block_list_by_index_stock(idx_stock)
        blk = blocks[0]
        all_stocks = blk.get_stock_list()

        # 构建数据
        candidates = []
        for stk in all_stocks:
            try:
                if is_st_stock(stk):
                    continue
                fin = stk.get_finance_info()
                liutong = float(fin["liutongguben"])

                # 获取该日期的收盘价
                k = stk.get_kdata(day_query)
                if not k or len(k) == 0:
                    continue
                close = float(k[-1].close)
                if close <= 0:
                    continue

                cap = liutong * close
                if cap > max_cap or cap <= 0:
                    continue
                candidates.append({
                    "code": stk.market_code,
                    "name": get_stock_display_name(stk),
                    "stock": stk,
                    "cap": cap,
                    "price": close,
                    "liutong": liutong,
                })
            except Exception:
                continue

        candidates.sort(key=lambda x: x["cap"])
        pool = candidates[:buy_count * pool_mult]
        if not pool:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "无候选股票"})

        # 获取前一日数据用于涨跌停过滤（回退10天足够跨越节假日）
        prev_query = Query(target_dt - Days(10), target_dt)
        bar_data = {}
        for item in pool:
            try:
                k = item["stock"].get_kdata(prev_query)
                if not k or len(k) < 2:
                    continue
                bar_data[item["code"]] = {
                    "prev_close": float(k[-2].close),
                    "close": float(k[-1].close),
                    "high": float(k[-1].high),
                    "low": float(k[-1].low),
                    "vol": float(k[-1].volume),
                }
            except Exception:
                continue

        buyable = []
        for item in pool:
            d = bar_data.get(item["code"])
            if not d:
                continue
            if d["high"] >= d["prev_close"] * 1.095:
                continue
            if d["low"] <= d["prev_close"] * 0.905:
                continue
            if d["vol"] == 0:
                continue
            buyable.append(item)

        signals = []
        buy_set = set(item["code"] for item in buyable[:buy_count])
        holdings_set = set(holdings or [])

        # 买入: 在目标列表中且不在当前持仓
        for item in buyable[:buy_count]:
            code = item["code"]
            weight = round(1.0 / min(len(buyable), buy_count), 4)
            is_held = code in holdings_set
            signals.append(StockSignal(
                code=code, name=item["name"],
                signal="hold" if is_held else "buy",
                reason=("已持仓" if is_held else f"流通市值第{[c['code'] for c in buyable[:buy_count]].index(code)+1}小 ({item['cap']/1e8:.2f}亿)"),
                market_cap=item["cap"], weight=weight, price=item["price"],
            ))

        # 卖出: 在持仓但不在目标列表中
        signals.extend(_make_sell_signals(holdings_set, [item["code"] for item in buyable[:buy_count]], day_query))

        metadata = {
            "total_candidates": len(candidates),
            "after_filter": len(buyable),
            "buy_count": len(signals),
            "trade_date": date_str,
            "index_name": "中小板指(399101)",
        }
        return StrategyResult(date=date_str, strategy_name=self.name,
                              signals=signals, metadata=metadata)


class MultiGoldenCrossStrategy(BaseStrategy):
    """多金叉共振 — MA金叉买入 + MA死叉卖出"""

    @property
    def name(self):
        return "多金叉共振"

    @property
    def description(self):
        return "一买一卖：MA5上穿MA20+放量时买入，MA5下穿MA20时卖出（中小板指成份股）"

    @property
    def params(self):
        return [
            StrategyParam("max_hold", "持仓数量", 5, "int", min_val=1, max_val=20),
            StrategyParam("ma_short", "短期均线", 5, "int", min_val=3, max_val=30),
            StrategyParam("ma_long", "长期均线", 20, "int", min_val=10, max_val=120),
            StrategyParam("vol_ratio", "买入量比", 1.5, "float", min_val=0.5, max_val=5.0, step=0.1),
        ]

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        max_hold = int(param_values.get("max_hold", 5))
        ma_s = int(param_values.get("ma_short", 5))
        ma_l = int(param_values.get("ma_long", 20))
        vol_r = float(param_values.get("vol_ratio", 1.5))

        if date is not None:
            target_dt = _to_hikyuu_date(date)
        else:
            _, _ds = _get_last_trade_date()
            target_dt = _to_hikyuu_date(_ds)
        date_str = _to_date_str(target_dt)
        if not sm.get_trading_calendar(Query(target_dt, target_dt + Days(1))):
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "当日非交易日或数据未更新"})

        look_q = Query(target_dt - Days(ma_l + 20), target_dt + Days(1))
        idx_stock = sm["sz399101"]
        blk = sm.get_block_list_by_index_stock(idx_stock)[0]
        all_stocks = blk.get_stock_list()

        def scan(stk):
            try:
                if is_st_stock(stk): return None
                k = stk.get_kdata(look_q)
                if not k or len(k) < ma_l + 5: return None
                closes = np.array([float(r.close) for r in k])
                vols = np.array([float(r.volume) for r in k])
                n = len(closes)
                ma_s_now, ma_l_now = np.mean(closes[-ma_s:]), np.mean(closes[-ma_l:])
                # 买入: MA5 在 MA20 上方（金叉状态，不要求当日刚金叉）
                if not (ma_s_now > ma_l_now): return None
                vol_ma = np.mean(vols[-ma_s:])
                if vol_ma <= 0 or vols[-1] / vol_ma < vol_r: return None
                return StockSignal(code=stk.market_code, name=get_stock_display_name(stk),
                                   signal="buy", reason="MA金叉", price=float(closes[-1]))
            except Exception: return None

        def check_sell(code):
            try:
                stk = sm[code[:2].lower() + code[2:]]
                k = stk.get_kdata(look_q)
                if not k or len(k) < ma_l + 3: return None
                closes = np.array([float(r.close) for r in k])
                n = len(closes)
                ms, ml = np.mean(closes[-ma_s:]), np.mean(closes[-ma_l:])
                # 卖出: MA5 在 MA20 下方（死叉状态）
                if ms < ml:
                    return StockSignal(code=code, name=get_stock_display_name(stk),
                                       signal="sell", reason="MA死叉", price=float(closes[-1]))
                return None
            except Exception: return None

        from concurrent.futures import ThreadPoolExecutor, as_completed
        buy_list, sell_list = [], []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for f in as_completed({pool.submit(scan, stk): stk for stk in all_stocks}):
                r = f.result()
                if r: buy_list.append(r)

        buy_list.sort(key=lambda x: x.price, reverse=True)
        top_buy = buy_list[:max_hold]
        buy_codes = {s.code for s in top_buy}
        sell_codes = set(holdings or []) - buy_codes

        with ThreadPoolExecutor(max_workers=8) as pool:
            for f in as_completed({pool.submit(check_sell, c): c for c in sell_codes}):
                r = f.result()
                if r: sell_list.append(r)

        signals = top_buy + sell_list
        if not signals:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "未选出"})

        return StrategyResult(
            date=date_str, strategy_name=self.name, signals=signals,
            metadata={"trade_date": date_str, "scan_count": len(all_stocks),
                      "buy": len(top_buy), "sell": len(sell_list)},
        )


# ── 启明星 ──
class MorningStarStrategy(BaseStrategy):
    """启明星 — 三根K线底部反转买入 + 趋势转弱卖出"""

    @property
    def name(self):
        return "启明星形态"

    @property
    def description(self):
        return "一买一卖：长阴→星线→长阳时买入，收盘跌破MA10时卖出（中小板指成份股）"

    @property
    def params(self):
        return [
            StrategyParam("max_hold", "持仓数量", 5, "int", min_val=1, max_val=20),
            StrategyParam("lookback_days", "搜索范围(天)", 15, "int", min_val=5, max_val=30),
            StrategyParam("small_body_ratio", "星线实体比例", 0.3, "float", min_val=0.1, max_val=0.8, step=0.05),
            StrategyParam("volume_ratio", "放量倍数", 1.5, "float", min_val=0.5, max_val=5.0, step=0.1),
        ]

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        max_hold = int(param_values.get("max_hold", 5))
        lb = int(param_values.get("lookback_days", 15))
        sbr = float(param_values.get("small_body_ratio", 0.3))
        vr = float(param_values.get("volume_ratio", 1.5))

        if date is not None:
            target_dt = _to_hikyuu_date(date)
        else:
            _, _ds = _get_last_trade_date()
            target_dt = _to_hikyuu_date(_ds)
        date_str = _to_date_str(target_dt)
        if not sm.get_trading_calendar(Query(target_dt, target_dt + Days(1))):
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "当日非交易日或数据未更新"})

        look_q = Query(target_dt - Days(lb + 20), target_dt + Days(1))
        idx_stock = sm["sz399101"]
        blk = sm.get_block_list_by_index_stock(idx_stock)[0]
        all_stocks = blk.get_stock_list()

        def scan(stk):
            try:
                if is_st_stock(stk): return None
                k = stk.get_kdata(look_q)
                if not k or len(k) < 6: return None
                r3, r2, r1 = k[-4], k[-3], k[-2]
                body3 = float(r3.close) - float(r3.open)
                if body3 >= 0: return None
                body3_size, range3 = abs(body3), float(r3.high) - float(r3.low)
                if range3 == 0 or body3_size / range3 < 0.3: return None
                if abs(float(r2.close) - float(r2.open)) > body3_size * sbr: return None
                if float(r1.close) - float(r1.open) <= 0: return None
                if float(r1.close) <= float(r3.open) * 1.01: return None
                vol_q = Query(target_dt - Days(lb + 20), target_dt + Days(1))
                kf = stk.get_kdata(vol_q)
                if kf and len(kf) > 5:
                    vs = [float(x.volume) for x in kf]
                    vm = np.mean(vs[-6:-1])
                    if vm > 0 and vs[-1] / vm < vr: return None
                return StockSignal(code=stk.market_code, name=get_stock_display_name(stk),
                                   signal="buy", reason="启明星形态", price=float(k[-2].close))
            except Exception: return None

        def check_sell(code):
            try:
                stk = sm[code[:2].lower() + code[2:]]
                k = stk.get_kdata(look_q)
                if not k or len(k) < 15: return None
                cs = [float(r.close) for r in k]
                # 卖出: MA5死叉MA10（比跌破MA10更稳定）
                ma5, ma10 = np.mean(cs[-5:]), np.mean(cs[-10:])
                ma5_prev = np.mean(cs[-6:-1])
                if ma5 < ma10 and ma5_prev >= ma10:
                    return StockSignal(code=code, name=get_stock_display_name(stk),
                                       signal="sell", reason="MA5死叉MA10", price=cs[-1])
                return None
            except Exception: return None

        from concurrent.futures import ThreadPoolExecutor, as_completed
        buy_list, sell_list = [], []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for f in as_completed({pool.submit(scan, stk): stk for stk in all_stocks}):
                r = f.result()
                if r: buy_list.append(r)

        buy_list.sort(key=lambda x: x.price)
        top_buy = buy_list[:max_hold]
        buy_codes = {s.code for s in top_buy}

        with ThreadPoolExecutor(max_workers=8) as pool:
            for f in as_completed({pool.submit(check_sell, c): c
                                   for c in (set(holdings or []) - buy_codes)}):
                r = f.result()
                if r: sell_list.append(r)

        signals = top_buy + sell_list
        if not signals:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "未选出启明星形态"})

        return StrategyResult(
            date=date_str, strategy_name=self.name, signals=signals,
            metadata={"trade_date": date_str, "buy": len(top_buy), "sell": len(sell_list)},
        )


# ── 科技动量轮动 ──
_STOCK_POOL = [
    "sz300308", "sz300502", "sz300394", "sz002281",
    "sz300620", "sh688498", "sh688313", "sz300548",
    "sz300570", "sh603083",
    "sh603986", "sh688525", "sz301308", "sz001309",
    "sh688008", "sz300475", "sh688110",
    "sh600118", "sh600879", "sz002025", "sh688270",
    "sz001270",
    "sz002384", "sz002475", "sz300454",
    "sz002407", "sh688981", "sz301217",
    "sh688560", "sh603601", "sh601869", "sh603220",
]


class TechMomentumStrategy(BaseStrategy):
    """科技动量轮动 — 固定科技股池动量打分轮动"""

    @property
    def name(self):
        return "科技动量轮动"

    @property
    def description(self):
        return "固定科技股池(光模块/芯片/航天)，动量打分TOP N等权买入"

    @property
    def params(self):
        return [
            StrategyParam("max_hold", "持仓数量", 5, "int", min_val=1, max_val=15),
            StrategyParam("lookback", "动量周期(天)", 20, "int", min_val=5, max_val=60),
            StrategyParam("top_n", "选股数量", 5, "int", min_val=1, max_val=15),
        ]

    def _score(self, stk, target_dt, lb):
        """动量得分：对数线性回归 + R² + 加速度 + 量比"""
        try:
            q = Query(target_dt - Days(lb * 2 + 10), target_dt + Days(1))
            k = stk.get_kdata(q)
            if not k or len(k) < lb:
                return 0
            c = np.array([float(r.close) for r in k])
            v = np.array([float(r.volume) for r in k])
            n = len(c)
            y = np.log(c); x = np.arange(n)
            w = np.linspace(1, 2, n)
            slope, _ = np.polyfit(x, y, 1, w=w)
            annual = np.exp(slope * 250) - 1
            ss_r = np.sum(w * (y - (slope*x + _))**2)
            ss_t = np.sum(w * (y - np.mean(y))**2)
            r2 = 1 - ss_r/ss_t if ss_t > 0 else 0
            score = annual * r2
            if score <= 0:
                return 0
            if n >= 5 and min(c[-4:]/c[-5:-1] - 1) < -0.03:
                score *= 0.5
            if n >= 6:
                a = c[-1]/c[-5] - 1
                if a > 0:
                    score *= (1 + a)
            if n >= 25:
                vr = (np.mean(v[-5:]) / np.mean(v[-20:-5]))
                if vr > 0:
                    score *= vr
            return score
        except Exception:
            return 0

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        max_hold = int(param_values.get("max_hold", 5))
        lb = int(param_values.get("lookback", 20))

        if date is not None:
            target_dt = _to_hikyuu_date(date)
        else:
            _, ds = _get_last_trade_date()
            target_dt = _to_hikyuu_date(ds)
        date_str = _to_date_str(target_dt)
        if not sm.get_trading_calendar(Query(target_dt, target_dt + Days(1))):
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "当日非交易日或数据未更新"})

        day_q = Query(target_dt, target_dt + Days(1))
        scored = []
        for code in _STOCK_POOL:
            try:
                stk = sm[code]
                s = self._score(stk, target_dt, lb)
                if s > 0:
                    k = stk.get_kdata(day_q)
                    px = float(k[-1].close) if k and len(k) > 0 else 0
                    scored.append((code, s, px))
            except Exception:
                continue
        if not scored:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "无有效得分"})

        scored.sort(key=lambda x: x[1], reverse=True)
        top_n = min(int(param_values.get("top_n", 5)), len(scored))
        top = scored[:top_n]
        buy_codes = {x[0] for x in top}
        holdings_set = set(holdings or [])

        signals = []
        for code, score, px in top:
            sig = "hold" if code in holdings_set else "buy"
            stk = sm[code]
            signals.append(StockSignal(
                code=code, name=get_stock_display_name(stk),
                signal=sig, reason=f"动量得分:{score:.2f}",
                price=px, weight=round(1.0/top_n, 3),
            ))

        for code in sorted(holdings_set - buy_codes):
            try:
                stk = sm[code]
                q = Query(target_dt - Days(40), target_dt + Days(1))
                k = stk.get_kdata(q)
                if not k or len(k) < 15:
                    continue
                cs = [float(r.close) for r in k]
                if cs[-1] < np.mean(cs[-10:]):
                    signals.append(StockSignal(
                        code=code, name=get_stock_display_name(stk),
                        signal="sell", reason="跌破MA10",
                        price=cs[-1],
                    ))
            except Exception:
                continue

        return StrategyResult(
            date=date_str, strategy_name=self.name, signals=signals,
            metadata={"trade_date": date_str, "scored": len(scored),
                      "top": top_n},
        )


# ── ETF双池平滑动量轮动 ──
class ETFMomentumStrategy(BaseStrategy):
    """ETF双池平滑动量轮动 — 静态+动态双池融合，加权平滑动量打分"""

    def __init__(self):
        self._last_pool_date = None
        self._cached_pool = None

    @property
    def name(self):
        return "ETF双池动量轮动"

    @property
    def description(self):
        return "全市场ETF双池(静态+动态)融合，加权对数回归动量打分，双均线趋势过滤，放量排雷，8%止损，防御ETF自动切换"

    @property
    def params(self):
        return [
            StrategyParam("buy_count", "持仓数量", 1, "int", min_val=1, max_val=10),
            StrategyParam("momentum_lookback", "动量周期(天)", 25, "int", min_val=10, max_val=60),
            StrategyParam("ma_short", "短期均线", 20, "int", min_val=5, max_val=60),
            StrategyParam("ma_long", "长期均线", 60, "int", min_val=20, max_val=120),
            StrategyParam("top_n", "候选池数量", 5, "int", min_val=1, max_val=20),
            StrategyParam("volume_ratio", "放量过滤阈值", 2.5, "float", min_val=1.0, max_val=5.0, step=0.1),
            StrategyParam("stop_loss", "止损比例(%)", 8, "float", min_val=3, max_val=20, step=1),
            StrategyParam("dynamic_pool_size", "动态池大小", 100, "int", min_val=30, max_val=300),
        ]

    def _weighted_log_regression_score(self, kdata, lookback=25):
        """
        加权对数平滑动量评分

        取最近 lookback 天收盘价:
          1. y = ln(close), x = arange(n)
          2. weights = linspace(1, 2, n) — 近期更高权重
          3. 加权 OLS 回归 → 斜率
          4. 年化收益 = exp(slope * 250) - 1
          5. 加权 R² = 1 - SS_res / SS_tot
          6. 得分 = max(0, min(5, 年化收益 * R²))
        """
        closes = np.array([float(r.close) for r in kdata])
        if len(closes) < lookback:
            return 0.0
        closes = closes[-lookback:]
        if np.any(closes <= 0):
            return 0.0

        y = np.log(closes)
        x = np.arange(len(closes))
        w = np.linspace(1.0, 2.0, len(closes))

        # 加权均值
        total_w = np.sum(w)
        x_mean = np.sum(w * x) / total_w
        y_mean = np.sum(w * y) / total_w

        # 加权协方差 / 方差
        wcov = np.sum(w * (x - x_mean) * (y - y_mean)) / total_w
        wvar = np.sum(w * (x - x_mean) ** 2) / total_w
        if wvar == 0:
            return 0.0

        slope = wcov / wvar
        intercept = y_mean - slope * x_mean

        annual_return = np.exp(slope * 250) - 1
        if annual_return <= 0:
            return 0.0

        # 加权 R²
        y_pred = slope * x + intercept
        ss_res = np.sum(w * (y - y_pred) ** 2)
        ss_tot = np.sum(w * (y - y_mean) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        score = annual_return * r2
        return max(0.0, min(5.0, score))

    def _build_dynamic_pool(self, target_dt, pool_size):
        """
        扫描全市场 ETF，按 5 日均量排序取前 pool_size 只。
        使用缓存（同一交易日复用结果）+ 线程池加速。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from datetime import datetime as dt_mod

        cache_key = _to_date_str(target_dt)
        if self._cached_pool is not None and self._last_pool_date == cache_key:
            return self._cached_pool

        # 查询足够覆盖 5 日均量的 K 线（以 target_dt 为截止，无未来函数）
        # 取 60 自然日 ≈ 40 交易日，足以计算 5 日均量
        from datetime import timedelta as tdelta
        if hasattr(target_dt, 'datetime'):
            py_vol_start = target_dt.datetime() - tdelta(days=60)
        else:
            py_vol_start = dt_mod(target_dt.year, target_dt.month, target_dt.day) - tdelta(days=60)
        hk_vol_start = _to_hikyuu_date(py_vol_start)
        vol_query = Query(hk_vol_start, target_dt)

        def _get_volume(stk):
            try:
                k = stk.get_kdata(vol_query)
                if not k or len(k) < 2:
                    return None
                vols = [float(r.volume) for r in k]
                # 只取最近 5 个有数据的交易日
                recent = [v for v in vols if v > 0][-5:]
                if len(recent) < 3:
                    return None
                avg_vol = np.mean(recent)
                return (stk.market_code, avg_vol)
            except Exception:
                return None

        # 收集所有 ETF
        all_etfs = [s for s in sm if s.type == constant.STOCKTYPE_ETF and s.valid]

        results = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            fut_map = {pool.submit(_get_volume, stk): stk for stk in all_etfs}
            for f in as_completed(fut_map):
                r = f.result()
                if r:
                    results.append(r)

        results.sort(key=lambda x: x[1], reverse=True)
        self._cached_pool = [code for code, _ in results[:pool_size]]
        self._last_pool_date = cache_key
        return self._cached_pool

    def _triple_filter(self, stk, target_dt, ma_short, ma_long, vol_ratio):
        """
        三重筛选（无未来函数）：
          1. 趋势: close > MA_short AND MA_short > MA_long
          2. 量价: 当日量 > 5日均量 × vol_ratio → 剔除
          3. 数据完整性: K线 >= ma_long + 10

        所有数据查询以 target_dt 为截止点，用自然日偏移确保数据量够。
        返回 (passed, kdata)
        """
        # 用自然日偏移（非交易日偏移）确保拿到足够历史数据
        # 250 自然日 ≈ 170 交易日，足够覆盖 MA60+ 所需
        from datetime import timedelta
        if hasattr(target_dt, 'datetime'):
            py_start = target_dt.datetime() - timedelta(days=300)
        else:
            py_start = datetime(target_dt.year, target_dt.month, target_dt.day) - timedelta(days=300)
        hk_start = _to_hikyuu_date(py_start)

        q = Query(hk_start, target_dt)  # 严格以 target_dt 为截止
        try:
            k = stk.get_kdata(q)
        except Exception:
            return False, None
        if not k or len(k) < ma_long + 5:
            return False, None

        closes = np.array([float(r.close) for r in k])
        volumes = np.array([float(r.volume) for r in k])

        # ── 趋势: close > MA_short and MA_short > MA_long ──
        ma_s = np.mean(closes[-ma_short:])
        ma_l = np.mean(closes[-ma_long:])
        if closes[-1] <= ma_s or ma_s <= ma_l:
            return False, None

        # ── 量价: 当日量 > 5日均量 × vol_ratio ──
        if len(volumes) >= 6:
            current_vol = volumes[-1]
            avg_5d = np.mean(volumes[-6:-1])
            if avg_5d > 0 and current_vol > avg_5d * vol_ratio:
                return False, None

        return True, k

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        from trade_advisor.etf_pool import (
            STATIC_POOL, DEFENSIVE_ETF_CODE, get_etf_category, get_etf_name as pool_get_name,
            CODE_TO_ETF,
        )

        buy_count = int(param_values.get("buy_count", 1))
        lookback = int(param_values.get("momentum_lookback", 25))
        ma_s = int(param_values.get("ma_short", 20))
        ma_l = int(param_values.get("ma_long", 60))
        top_n = int(param_values.get("top_n", 5))
        vol_ratio = float(param_values.get("volume_ratio", 2.5))
        dyn_pool_size = int(param_values.get("dynamic_pool_size", 100))
        stop_loss_pct = float(param_values.get("stop_loss", 8))

        # ── 1. 日期校验 ──
        if date is not None:
            target_dt = _to_hikyuu_date(date)
        else:
            _, ds = _get_last_trade_date()
            if ds == "N/A":
                return StrategyResult(date="N/A", strategy_name=self.name,
                                      metadata={"error": "无法获取交易日历"})
            target_dt = _to_hikyuu_date(ds)
        date_str = _to_date_str(target_dt)

        cal = sm.get_trading_calendar(Query(target_dt, target_dt + Days(1)))
        if not cal or len(cal) == 0:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "当日非交易日或数据未更新"})

        # ── 数据存在性检查 ──
        # 交易日历通过不代表数据已下载，用参考ETF验证实际K线可用性
        _ref_stock = sm["sh510050"]
        _ref_q = Query(target_dt, target_dt + Days(1))
        _ref_k = _ref_stock.get_kdata(_ref_q)
        _has_data = (_ref_k is not None and len(_ref_k) > 0 and float(_ref_k[-1].close) > 0)

        if not _has_data:
            # 数据未更新到所选日期，自动回退到最近有数据的交易日
            _, _fallback_ds = _get_last_trade_date()
            if _fallback_ds and _fallback_ds != "N/A":
                import logging
                logging.warning(
                    f"ETF动量: {date_str} 无数据，自动回退到 {_fallback_ds}"
                )
                target_dt = _to_hikyuu_date(_fallback_ds)
                date_str = _fallback_ds
            else:
                return StrategyResult(date=date_str, strategy_name=self.name,
                                      metadata={"error": f"{date_str} 无K线数据，且无法找到最近有数据的交易日"})

        # ── 2. 构建融合池 ──
        # 静态池 code 集合
        static_codes = {e.code for e in STATIC_POOL if e.code != DEFENSIVE_ETF_CODE}

        # 动态池（全市场ETF按5日均量排序）
        try:
            dynamic_codes = self._build_dynamic_pool(target_dt, dyn_pool_size)
        except Exception:
            dynamic_codes = []

        # 融合池 = 静态 ∪ 动态
        merged_codes = list(static_codes | set(dynamic_codes))

        # ── 3. 对每只 ETF 执行三重筛选 + 动量评分 ──
        from concurrent.futures import ThreadPoolExecutor, as_completed

        candidates = []

        def evaluate(code):
            """单只 ETF 评估：筛选 + 评分"""
            try:
                stk = sm[code.lower()]
                is_defensive = (code == DEFENSIVE_ETF_CODE)

                # 价格必须从 date-specific Query 获取（评分用负偏移Query会取到最新数据）
                day_q = Query(target_dt, target_dt + Days(1))
                k_day = stk.get_kdata(day_q)
                close = float(k_day[-1].close) if k_day and len(k_day) > 0 else 0
                if close <= 0:
                    return None

                if is_defensive:
                    score = 0.01
                    price = close
                else:
                    passed, kdata = self._triple_filter(stk, target_dt, ma_s, ma_l, vol_ratio)
                    if not passed or kdata is None:
                        return None
                    score = self._weighted_log_regression_score(kdata, lookback)
                    if score <= 0:
                        return None
                    price = close

                name = stk.name
                try:
                    name = name.encode("latin-1").decode("gbk", errors="replace")
                except Exception:
                    pass

                # 尝试从静态池获取更友好的名称
                pool_name = pool_get_name(code)
                if pool_name != code:
                    name = pool_name

                category = get_etf_category(code)
                return {
                    "code": code,
                    "name": name,
                    "score": score,
                    "price": price,
                    "category": category,
                    "defensive": is_defensive,
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            fut_map = {pool.submit(evaluate, code): code for code in merged_codes}
            for f in as_completed(fut_map):
                r = f.result()
                if r:
                    candidates.append(r)

        if not candidates:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "无任何候选ETF通过筛选", "total_scanned": len(merged_codes)})

        # ── 4. 排序选 Top N ──
        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected = []
        categories_used = set()
        for c in candidates:
            if len(selected) >= top_n:
                break
            # 行业分散：同分类限 1 只（防御ETF除外）
            if c["category"] != "未知" and c["category"] in categories_used and not c["defensive"] and buy_count > 1:
                continue
            categories_used.add(c["category"])
            selected.append(c)

        # 如果选出的候选不足，放宽分类限制再补选
        if len(selected) < top_n:
            for c in candidates:
                if len(selected) >= top_n:
                    break
                    # 跳过防御ETF（只在无任何候选时才用）
                if c["defensive"] and len(selected) > 0:
                    continue
                if c not in selected:
                    selected.append(c)

        # ── 5. 防御检查：无候选或全部得分过低 → 切到防御ETF ──
        if not selected or all(c["score"] < 0.001 and not c["defensive"] for c in selected[:buy_count]):
            # 尝试加入防御ETF
            def_etf_eval = evaluate(DEFENSIVE_ETF_CODE)
            if def_etf_eval and def_etf_eval["price"] > 0:
                selected = [def_etf_eval]
                buy_count = 1
                defensive_mode = True
            else:
                defensive_mode = False
        else:
            defensive_mode = False

        # ── 6. 生成信号 ──
        signals = []
        buy_set = set()
        holdings_set = set(holdings or [])

        for item in selected[:buy_count]:
            code = item["code"]
            buy_set.add(code)
            is_held = code in holdings_set
            weight = round(1.0 / min(len(selected[:buy_count]), buy_count), 4)
            sig = "hold" if is_held else "buy"
            reason_parts = []
            if defensive_mode:
                reason_parts.append("防御模式-无候选")
            else:
                reason_parts.append(f"动量得分:{item['score']:.3f}")
            reason_parts.append(f"分类:{item['category']}")
            signals.append(StockSignal(
                code=code, name=item["name"],
                signal=sig, reason=" | ".join(reason_parts),
                market_cap=0, weight=weight, price=item["price"],
            ))

        # 卖出：持仓不在目标列表或有止损/放量信号
        from trade_advisor.etf_pool import get_etf_name as fallback_name_2
        for code in sorted(holdings_set - buy_set):
            try:
                stk = sm[code.lower()]
                name = stk.name
                try:
                    name = name.encode("latin-1").decode("gbk", errors="replace")
                except Exception:
                    pass
                fb_name = fallback_name_2(code)
                if fb_name != code:
                    name = fb_name
                q = Query(target_dt, target_dt + Days(1))
                k = stk.get_kdata(q)
                price = float(k[-1].close) if k and len(k) > 0 else 0
                signals.append(StockSignal(
                    code=code, name=name,
                    signal="sell", reason="调出目标列表",
                    market_cap=0, price=price, weight=0,
                ))
            except Exception:
                signals.append(StockSignal(
                    code=code, name=code,
                    signal="sell", reason="调出目标列表",
                    price=0, weight=0,
                ))

        # ── 7. metadata ──
        metadata = {
            "trade_date": date_str,
            "static_pool": len(static_codes),
            "dynamic_pool": len(dynamic_codes),
            "merged_pool": len(merged_codes),
            "passed_filter": len(candidates),
            "top_n": top_n,
            "buy_count": min(len(selected), buy_count),
            "defensive_mode": defensive_mode,
        }

        return StrategyResult(date=date_str, strategy_name=self.name,
                              signals=signals, metadata=metadata)


# 策略注册表
REGISTERED_STRATEGIES = {
    # 原有策略
    "small_cap": SmallCapStrategy(),
    "golden_cross": MultiGoldenCrossStrategy(),
    "morning_star": MorningStarStrategy(),
    "tech_momentum": TechMomentumStrategy(),
    "etf_momentum": ETFMomentumStrategy(),
}

# 延迟导入新策略（避免循环导入）
def _register_khunter_strategies():
    """注册来自 KHunter 迁移的策略（延迟导入）"""
    try:
        from trade_advisor.strategies.khunter_golden_cross import MultiGoldenCrossStrategyKH
        from trade_advisor.strategies.w_bottom import WBottomStrategy
        from trade_advisor.strategies.limit_up_pullback import LimitUpPullbackStrategy

        REGISTERED_STRATEGIES.update({
            "golden_cross_kh": MultiGoldenCrossStrategyKH(),
            "w_bottom": WBottomStrategy(),
            "limit_up_pullback": LimitUpPullbackStrategy(),
        })
        return True
    except ImportError as e:
        # 新策略未就绪时，仅使用原有策略
        import logging
        logging.warning(f"KHunter 策略导入失败，仅使用原有策略: {e}")
        return False


def _register_trend_momentum_strategy():
    """注册趋势动量因子策略"""
    try:
        from trade_advisor.strategies.trend_momentum import TrendMomentumFactorStrategy
        REGISTERED_STRATEGIES["trend_momentum"] = TrendMomentumFactorStrategy()
        return True
    except ImportError as e:
        import logging
        logging.warning(f"趋势动量因子策略导入失败: {e}")
        return False


def _register_enhanced_multifactor():
    """注册增强版多因子策略"""
    try:
        from trade_advisor.strategies.enhanced_multifactor import EnhancedMultiFactorStrategy
        REGISTERED_STRATEGIES["enhanced_multifactor"] = EnhancedMultiFactorStrategy()
        return True
    except ImportError as e:
        import logging
        logging.warning(f"增强多因子策略导入失败: {e}")
        return False


def _register_ml_strategy():
    """注册 ML 因子策略"""
    try:
        from trade_advisor.strategies.ml_strategy import MLFactorStrategy
        REGISTERED_STRATEGIES["ml_factor"] = MLFactorStrategy()
        return True
    except ImportError as e:
        import logging
        logging.warning(f"ML 因子策略导入失败: {e}")
        return False


# 在模块加载时注册
_register_khunter_strategies()
_register_trend_momentum_strategy()
_register_enhanced_multifactor()
_register_ml_strategy()


