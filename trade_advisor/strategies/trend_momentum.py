"""
趋势动量因子策略 — 可回测、可优化的量化评分模型

因子体系（总分 100）：
  20%  涨幅因子    — 3~5% 温和上涨
  20%  量能因子    — 量比 > 1.5，资金介入
  20%  均线趋势因子 — MA5 > MA10 > MA20，短中期多头
  15%  换手因子    — 5~10% 筹码充分交换
  15%  新高因子    — 收盘价突破20日新高
  10%  板块强度因子 — 所属板块排名前10

交易规则：
  买入 — 评分 >= 70，全市场选前 N 只
  卖出 — 收盘跌破 MA10（趋势终结信号）
  仓位 — 等权分配，每日调仓

设计思路：
  - 不做 "条件 and 条件" 的硬过滤，而是因子加权评分
  - 每个因子独立可调，便于回测优化权重
  - 可扩展：增加/替换因子只需修改因子列表
"""

import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

from hikyuu.interactive import *
from trade_advisor.strategies import (
    BaseStrategy, StrategyParam, StockSignal, StrategyResult,
    _to_hikyuu_date, _to_date_str, _get_last_trade_date, _make_sell_signals,
    get_stock_display_name, is_st_stock,
)
from trade_advisor.data_adapter import get_kdata, get_stock_list


# ── 因子定义 ──
FACTOR_DEFS = [
    {"name": "涨幅",     "key": "pct_score",     "weight": 20, "max": 20},
    {"name": "量能",     "key": "vol_score",      "weight": 20, "max": 20},
    {"name": "均线趋势", "key": "ma_score",       "weight": 20, "max": 20},
    {"name": "换手",     "key": "turnover_score", "weight": 15, "max": 15},
    {"name": "新高",     "key": "high_score",     "weight": 15, "max": 15},
    {"name": "板块强度", "key": "sector_score",   "weight": 10, "max": 10},
]


class TrendMomentumFactorStrategy(BaseStrategy):
    """趋势动量因子策略 — 多因子加权评分选股"""

    def __init__(self):
        super().__init__()
        self._sector_cache = {}    # 板块强度缓存
        self._cache_date = None

    @property
    def name(self) -> str:
        return "趋势动量因子"

    @property
    def description(self) -> str:
        return "六因子加权评分：涨幅+量能+均线+换手+新高+板块强度，评分>=70买入，跌破MA10卖出"

    @property
    def params(self) -> list:
        return [
            StrategyParam("buy_count",     "持仓数量",    5,  "int",   min_val=1,   max_val=20),
            StrategyParam("score_threshold","买入阈值",   30, "int",   min_val=10,  max_val=90),
            StrategyParam("lookback_days",  "回溯天数",   60, "int",   min_val=30,  max_val=120),
            StrategyParam("pct_min",       "涨幅下限(%)", 1.5,"float", min_val=0,   max_val=10, step=0.5),
            StrategyParam("pct_max",       "涨幅上限(%)", 5.0,"float", min_val=2,   max_val=12, step=0.5),
            StrategyParam("vol_ratio_min", "量比下限",   1.0,"float", min_val=0.5,  max_val=5,  step=0.1),
            StrategyParam("turnover_min",  "换手下限(%)", 2,  "float", min_val=1,   max_val=20, step=1),
            StrategyParam("turnover_max",  "换手上限(%)", 12, "float", min_val=5,   max_val=30, step=1),
            StrategyParam("float_mv_min",  "市值下限(亿)",20, "float", min_val=5,   max_val=500, step=5),
            StrategyParam("float_mv_max",  "市值上限(亿)",2000,"float", min_val=30,  max_val=10000, step=10),
        ]

    # ================================================================
    # 因子计算（单只股票）
    # ================================================================

    def _calc_factors(self, df: pd.DataFrame, params: dict,
                      turnover: Optional[float] = None) -> Dict[str, float]:
        """
        对一只股票计算所有因子得分

        Args:
            df: K线DataFrame（升序），含 open/high/low/close/volume
            params: 策略参数字典

        Returns:
            {因子key: 得分, ...} 以及元数据
        """
        if df.empty or len(df) < 25:
            return {}

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_ = df['open'].values

        latest_close = close[-1]
        latest_vol = volume[-1]

        # ── 前置检查：最低数据量 ──
        if len(close) < 25 or latest_close <= 0:
            return {}

        # ── 1. 涨幅因子 ──
        pct_min = params.get("pct_min", 3.0)
        pct_max = params.get("pct_max", 5.0)
        prev_close = close[-2] if len(close) >= 2 else close[-1]
        pct_change = (latest_close - prev_close) / prev_close * 100

        pct_score = 0
        if pct_min <= pct_change <= pct_max:
            # 理想区间满分
            pct_score = 20
        elif 0 < pct_change < pct_min:
            # 线性递增：0%→0分，pct_min→20分
            pct_score = (pct_change / pct_min) * 20
        elif pct_change < 0:
            # 下跌不给分
            pct_score = 0
        else:
            # 超过上限，逐渐衰减
            overshoot = (pct_change - pct_max) / pct_max
            pct_score = max(0, 20 * (1 - overshoot))

        # ── 2. 量能因子 ──
        vol_ratio_min = params.get("vol_ratio_min", 1.5)
        vol_ma5 = np.mean(volume[-6:-1]) if len(volume) >= 6 else np.mean(volume)
        vol_ratio = latest_vol / vol_ma5 if vol_ma5 > 0 else 0

        vol_score = 0
        if vol_ratio > vol_ratio_min:
            # 线性：vol_ratio_min→1分, 2.5以上满分
            vol_score = min(20, 10 + (vol_ratio - 1.0) * 10)
        elif vol_ratio > 0.5:
            # 缩量但没完全缩死
            vol_score = 5

        # ── 3. 均线趋势因子 ──
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:]) if len(close) >= 10 else ma5
        ma20 = np.mean(close[-20:]) if len(close) >= 20 else ma10
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20

        ma_score = 0
        # 多头排列评分（逐层累加）
        if ma10 > ma20:
            ma_score += 5  # 中期趋势向上
        if ma5 > ma10:
            ma_score += 5  # 短期趋势向上
        if ma5 > ma10 > ma20:
            ma_score += 5  # 完全多头排列
        # 价格位置评分
        if latest_close > ma5:
            ma_score += 5  # 价格在短线上方
        elif latest_close > ma10:
            ma_score += 3  # 价格在中线上方

        # ── 4. 换手因子 ──
        # 换手率从 Hikyuu 财务数据计算后传入
        turnover_min = params.get("turnover_min", 5)
        turnover_max = params.get("turnover_max", 10)

        turnover_score = 0
        if turnover is not None and turnover > 0:
            if turnover_min <= turnover <= turnover_max:
                turnover_score = 15
            elif turnover < turnover_min:
                turnover_score = 7
            elif turnover <= turnover_max * 1.5:
                turnover_score = 7
        turnover_score = 0
        if turnover is not None:
            if turnover_min <= turnover <= turnover_max:
                turnover_score = 15
            elif turnover < turnover_min:
                turnover_score = 7
            elif turnover <= turnover_max * 1.5:
                turnover_score = 7

        # ── 5. 新高因子 ──
        high_20 = np.max(high[-21:-1]) if len(high) >= 22 else np.max(high[:-1])
        high_score = 0
        if latest_close > high_20:
            high_score = 15  # 突破新高
        elif high_20 > 0 and latest_close > high_20 * 0.95:
            # 接近新高 - 线性评分
            ratio = (latest_close - high_20 * 0.95) / (high_20 * 0.05)
            high_score = max(0, min(15, ratio * 15))

        # ── 6. 板块强度因子 ──
        sector_score = 0  # 由外部设置

        return {
            "pct_score":     pct_score,
            "vol_score":     vol_score,
            "ma_score":      ma_score,
            "turnover_score": turnover_score,
            "high_score":    high_score,
            "sector_score":  sector_score,

            # 原始数据（用于调试和元数据）
            "_pct":       round(pct_change, 2),
            "_vol_ratio": round(vol_ratio, 2),
            "_turnover":  round(turnover, 2) if turnover else None,
            "_ma5":       round(ma5, 2),
            "_ma10":      round(ma10, 2),
            "_ma20":      round(ma20, 2),
            "_close":     round(latest_close, 2),
        }

    # ================================================================
    # 选股主逻辑
    # ================================================================

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        buy_count = int(param_values.get("buy_count", 5))
        score_threshold = int(param_values.get("score_threshold", 70))
        lookback_days = int(param_values.get("lookback_days", 60))
        float_mv_min = float(param_values.get("float_mv_min", 30)) * 1e8
        float_mv_max = float(param_values.get("float_mv_max", 200)) * 1e8

        # ── 日期 ──
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
                                  metadata={"error": "当日非交易日"})

        # ── 数据存在性检查 ──
        _ref = sm["sh510050"].get_kdata(Query(target_dt, target_dt + Days(1)))
        if not _ref or len(_ref) == 0 or float(_ref[-1].close) <= 0:
            _, fb = _get_last_trade_date()
            if fb and fb != "N/A":
                target_dt = _to_hikyuu_date(fb)
                date_str = fb
            else:
                return StrategyResult(date=date_str, strategy_name=self.name,
                                     metadata={"error": f"{date_str} 无K线数据"})

        # ── 获取股票池（全市场 A 股） ──
        try:
            stock_codes = get_stock_list('all')
            if not stock_codes:
                return StrategyResult(date=date_str, strategy_name=self.name,
                                     metadata={"error": "无法获取股票列表"})
        except Exception as e:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                 metadata={"error": f"股票列表获取失败: {e}"})

        # ── 扫描选股 ──
        query_start = (target_dt - Days(lookback_days + 20)).datetime().date()
        query_end = target_dt.datetime().date()

        candidates = []

        def scan_stock(code: str) -> Optional[Dict]:
            try:
                # 跳过 ETF、指数等
                if 'sh5' in code.lower() or 'sz3' in code.lower():
                    return None

                # 获取 K 线
                df = get_kdata(code, str(query_start), str(query_end))
                if df.empty or len(df) < 25:
                    return None

                # 获取财务数据和基础信息
                try:
                    stk = sm[code.lower()]
                    name = get_stock_display_name(stk)
                    if is_st_stock(stk):
                        return None

                    # 获取流通市值和换手率
                    fin = stk.get_finance_info()
                    close_price = float(df['close'].values[-1])
                    volume = float(df['volume'].values[-1])

                    turnover = None
                    float_mv = -1

                    if fin and fin.have('liutongguben'):
                        liutongguben = float(fin['liutongguben'])  # 流通股本（股）
                        # 流通市值（亿元）= 收盘价 * 流通股本 / 1亿
                        float_mv = close_price * liutongguben / 1e8

                        # 换手率（%）= 当日成交量(股) / 流通股本(股) * 100
                        if liutongguben > 0:
                            turnover = volume / liutongguben * 100
                except:
                    name = code
                    float_mv = -1
                    turnover = None

                # 市值过滤
                if float_mv > 0:
                    mv_yi = float_mv
                    if mv_yi < float_mv_min / 1e8 or mv_yi > float_mv_max / 1e8:
                        return None
                else:
                    # 无法获取市值时跳过市值过滤
                    pass

                # 计算因子（传入换手率）
                factors = self._calc_factors(df, param_values, turnover=turnover)
                if not factors:
                    return None

                # 计算总分
                total = sum(factors.get(k, 0) for k in
                           ["pct_score", "vol_score", "ma_score",
                            "turnover_score", "high_score", "sector_score"])

                if total < score_threshold:
                    return None

                return {
                    "code": code,
                    "name": name,
                    "score": total,
                    "price": factors["_close"],
                    "factors": factors,
                    "float_mv": float_mv,
                }

            except Exception:
                return None

        # 顺序扫描（Hikyuu C++ 后端非线程安全，不用 ThreadPoolExecutor）
        max_scan = min(1500, len(stock_codes))
        for i, code in enumerate(stock_codes[:max_scan]):
            if i % 100 == 0 and i > 0:
                logger.info(f"  扫描进度: {i}/{max_scan}")
            r = scan_stock(code)
            if r:
                candidates.append(r)
                if len(candidates) >= buy_count * 3:  # 有足够候选就提前退出
                    break

        # ── 排序选 Top ──
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:buy_count]

        if not selected:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  signals=[],
                                  metadata={"error": f"无股票评分>={score_threshold}",
                                            "scanned": len(stock_codes)})

        # ── 生成信号 ──
        signals = []
        buy_codes = set()
        for item in selected:
            code = item["code"]
            buy_codes.add(code)
            factor_detail = "; ".join(
                f"{k}:{item['factors'][k]}" for k in
                ["pct_score", "vol_score", "ma_score", "turnover_score", "high_score"]
                if item['factors'].get(k, 0) > 0
            )
            reason = (f"总分:{item['score']} | {factor_detail}"
                     if factor_detail else f"总分:{item['score']}")

            signals.append(StockSignal(
                code=code, name=item["name"],
                signal="buy", reason=reason,
                market_cap=item.get("float_mv", 0) * 1e8 if item.get("float_mv", 0) > 0 else 0,
                weight=round(1.0 / len(selected), 4),
                price=item["price"],
            ))

        # ── 卖出信号：跌破 MA10 ──
        if holdings:
            holdings_set = set(holdings)
            hold_to_sell = holdings_set - buy_codes
            for code in sorted(hold_to_sell):
                try:
                    stk = sm[code[:2].lower() + code[2:]]
                    k = stk.get_kdata(Query(target_dt - Days(30), target_dt + Days(1)))
                    if k and len(k) >= 15:
                        closes = np.array([float(r.close) for r in k])
                        ma10 = np.mean(closes[-10:])
                        if closes[-1] < ma10:
                            signals.append(StockSignal(
                                code=code, name=get_stock_display_name(stk),
                                signal="sell", reason=f"跌破MA10({ma10:.2f})",
                                price=float(closes[-1]),
                            ))
                        else:
                            # 在池中但未跌破 MA10 → hold
                            signals.append(StockSignal(
                                code=code, name=get_stock_display_name(stk),
                                signal="hold", reason=f"持仓中 MA10({ma10:.2f})",
                                price=float(closes[-1]),
                            ))
                    else:
                        signals.append(StockSignal(
                            code=code, name=code,
                            signal="sell", reason="数据不足，强制卖出",
                        ))
                except:
                    signals.append(StockSignal(
                        code=code, name=code,
                        signal="sell", reason="卖出",
                    ))

        # ── 元数据 ──
        avg_score = np.mean([s["score"] for s in selected]) if selected else 0
        factor_breakdown = {}
        for k in ["pct_score", "vol_score", "ma_score", "turnover_score", "high_score"]:
            vals = [s["factors"].get(k, 0) for s in selected]
            factor_breakdown[k] = round(np.mean(vals), 1) if vals else 0

        metadata = {
            "buy_count": len(selected),
            "avg_score": round(avg_score, 1),
            "scanned": len(stock_codes),
            "candidates": len(candidates),
            "factors": factor_breakdown,
        }

        return StrategyResult(date=date_str, strategy_name=self.name,
                              signals=signals, metadata=metadata)


# ================================================================
# 回测专用版本（简化、快速）
# ================================================================

# 为回测引擎提供的辅助函数：判断单日买卖信号
def backtest_signal(kdata, params: dict) -> str:
    """
    回测专用——根据 KData 判断当日是买(buy)/卖(sell)/空(hold)

    买入条件:
        - 评分 >= score_threshold
        - 收盘价 > MA5
        - 量比 > vol_ratio_min

    卖出条件:
        - 收盘价 < MA10

    Args:
        kdata: Hikyuu KData
        params: 策略参数

    Returns:
        'buy', 'sell', 或 ''
    """
    if not kdata or len(kdata) < 25:
        return ''

    closes = np.array([float(r.close) for r in kdata])
    volumes = np.array([float(r.volume) for r in kdata])
    highs = np.array([float(r.high) for r in kdata])

    c = closes[-1]          # 今日收盘
    vol = volumes[-1]        # 今日量
    prev_close = closes[-2] if len(closes) >= 2 else c
    pct = (c - prev_close) / prev_close * 100

    # 均线
    ma5 = np.mean(closes[-5:])
    ma10 = np.mean(closes[-10:])
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else ma10

    # 量比
    vol_ma5 = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else np.mean(volumes)
    vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 0

    # 新高
    high_20 = np.max(highs[-21:-1]) if len(highs) >= 22 else np.max(highs[:-1])

    # 参数
    pct_min = params.get("pct_min", 3.0)
    pct_max = params.get("pct_max", 5.0)
    vr_min = params.get("vol_ratio_min", 1.5)
    threshold = params.get("score_threshold", 70)

    # ── 快速评分 ──
    score = 0
    if pct_min <= pct <= pct_max:
        score += 20
    elif 0 < pct < pct_min:
        score += (pct / pct_min) * 20
    if vol_ratio > 1.5:
        score += 20
    elif vol_ratio > 1.0:
        score += 10
    if ma5 > ma10 > ma20 and c > ma5:
        score += 20
    if c > high_20:
        score += 15

    # ── 判断 ──
    if score >= threshold:
        return 'buy'
    elif c < ma10 and len(closes) >= 10:
        return 'sell'

    return ''
