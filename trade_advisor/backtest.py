"""
回测引擎 — 对策略进行历史回测并计算绩效指标
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

from hikyuu.interactive import *
from trade_advisor.strategies import (
    SmallCapStrategy, REGISTERED_STRATEGIES,
    _to_hikyuu_date, _to_date_str, _get_last_trade_date,
    StockSignal, StrategyResult,
)


@dataclass
class BacktestTrade:
    """一笔模拟交易"""
    date: str
    code: str
    name: str
    action: str          # buy / sell
    price: float
    shares: int
    value: float
    reason: str = ""
    holdings_count: int = 0    # 交易后持仓只数
    cash_after: float = 0      # 交易后现金余额


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    start_date: str
    end_date: str
    init_cash: float
    final_value: float
    total_return_pct: float
    annual_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trade_count: int
    trading_days: int
    nav_series: pd.Series = None        # 日期 -> 净值
    drawdown_series: pd.Series = None   # 日期 -> 回撤
    monthly_returns: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)


def run_backtest(
    strategy_key: str = "small_cap",
    start_date: str = None,
    end_date: str = None,
    params: dict = None,
    init_cash: float = 100_000,
    commission_rate: float = 0.0003,
    stamp_rate: float = 0.001,
    min_commission: float = 5.0,
    rebalance_days: int = 1,
) -> BacktestResult:
    """
    对指定策略执行回测

    参数:
        strategy_key: 策略标识
        start_date: 起始日期 (YYYY-MM-DD), 默认1年前
        end_date: 结束日期 (YYYY-MM-DD), 默认最新交易日
        params: 策略参数
        init_cash: 初始资金
        rebalance_days: 调仓周期（交易日数）, 1=每日, 5=每周, 20=每月
    """
    strategy = REGISTERED_STRATEGIES.get(strategy_key)
    if not strategy:
        raise ValueError(f"未知策略: {strategy_key}")

    params = params or {}
    buy_count = int(params.get("buy_count", 5))

    # 确定日期范围
    if end_date is None:
        _, end_str = _get_last_trade_date()
    else:
        end_str = end_date
    if start_date is None:
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=365)
        start_str = start_dt.strftime("%Y-%m-%d")
    else:
        start_str = start_date

    hk_start = _to_hikyuu_date(start_str)
    hk_end = _to_hikyuu_date(end_str)

    # 获取交易日历
    q = Query(hk_start, hk_end)
    cal = sm.get_trading_calendar(q)
    if not cal or len(cal) < 10:
        raise ValueError(f"交易日不足(共{len(cal) if cal else 0}天)，无法回测")

    trade_dates = [d.datetime() if hasattr(d, 'datetime') else d for d in cal]
    trade_dates = [d for d in trade_dates if start_str <= d.strftime("%Y-%m-%d") <= end_str]

    # 对 small_cap 策略做特殊优化：预加载价格矩阵
    if strategy_key == "small_cap":
        return _backtest_small_cap(
            trade_dates, params, init_cash,
            commission_rate, stamp_rate, min_commission,
            rebalance_days,
        )
    elif strategy_key == "etf_momentum":
        return _backtest_etf_momentum(
            trade_dates, params, init_cash,
            commission_rate, stamp_rate, min_commission,
            rebalance_days,
        )
    else:
        # 通用回测（含调仓周期支持）
        return _backtest_generic(
            strategy, trade_dates, params, init_cash,
            commission_rate, stamp_rate, min_commission,
            rebalance_days,
        )


def _backtest_small_cap(
    trade_dates, params, init_cash,
    commission_rate, stamp_rate, min_commission,
    rebalance_days=1,
):
    """小市值策略专用回测（使用预加载的价格矩阵加速）"""
    buy_count = int(params.get("buy_count", 5))
    pool_mult = int(params.get("pool_size", 3))
    max_cap = float(params.get("max_cap", 50)) * 1e8

    print(f"回测区间: {trade_dates[0].strftime('%Y-%m-%d')} ~ {trade_dates[-1].strftime('%Y-%m-%d')}")
    print(f"交易日数: {len(trade_dates)}, 调仓周期: {rebalance_days}天")

    # 获取股票池
    idx_stock = sm["sz399101"]
    blocks = sm.get_block_list_by_index_stock(idx_stock)
    blk = blocks[0]
    all_stocks = blk.get_stock_list()

    # 预加载价格数据
    price_data = {}   # code -> {date -> close}
    name_map = {}
    liutong_map = {}

    from trade_advisor.strategies import get_stock_display_name, is_st_stock

    for stk in all_stocks:
        try:
            if is_st_stock(stk):
                continue
            code = stk.market_code
            name_map[code] = get_stock_display_name(stk)
            fin = stk.get_finance_info()
            liutong_map[code] = float(fin["liutongguben"])

            # 预加载所有日线数据
            k = stk.get_kdata(Query(-len(trade_dates) - 20))
            if not k or len(k) == 0:
                continue
            prices = {}
            for r in k:
                dt = r.datetime.datetime() if hasattr(r.datetime, 'datetime') else r.datetime
                ds = dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt)[:10]
                prices[ds] = float(r.close)
            if prices:
                price_data[code] = prices
        except Exception:
            continue

    valid_codes = [c for c in price_data if c in liutong_map and liutong_map[c] > 0]
    print(f"有效股票: {len(valid_codes)}")

    # 回测循环
    cash = float(init_cash)
    positions = {}          # code -> {shares, buy_price}
    daily_nav = []
    trades = []

    for di, date in enumerate(trade_dates):
        ds = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]

        # 计算当日持仓市值
        pos_value = 0
        for code, pos in list(positions.items()):
            px = price_data.get(code, {}).get(ds)
            if px and px > 0:
                pos["current_price"] = px
                pos_value += px * pos["shares"]
            else:
                pos_value += pos.get("buy_price", 0) * pos["shares"]

        total_value = cash + pos_value
        daily_nav.append({"date": ds, "total_value": total_value, "cash": cash})

        # 判断是否调仓日：按日历天数计算间隔
        if di == 0:
            is_rebalance = True
            last_rebalance_date = date
        else:
            days_since = (date - last_rebalance_date).days
            is_rebalance = days_since >= rebalance_days
        if not is_rebalance:
            continue
        # 更新上次调仓日期
        if di > 0:
            last_rebalance_date = date

        if di == 0:
            prev_ds = ds
        else:
            prev_ds = trade_dates[di - 1].strftime("%Y-%m-%d") if hasattr(trade_dates[di-1], 'strftime') else str(trade_dates[di-1])[:10]

        # 计算流通市值
        caps = {}
        for code in valid_codes:
            px = price_data.get(code, {}).get(ds)
            if px and px > 0:
                cap = liutong_map[code] * px
                if cap <= max_cap:
                    caps[code] = cap

        sorted_caps = sorted(caps.items(), key=lambda x: x[1])
        candidates = [c for c, _ in sorted_caps[:buy_count * pool_mult]]

        # 过滤涨跌停/停牌（第一天不过滤，因为没有前一天数据）
        buyable = []
        for code in candidates:
            px = price_data.get(code, {}).get(ds)
            prev_px = price_data.get(code, {}).get(prev_ds)
            if not px or px <= 0:
                continue
            if di > 0 and (not prev_px or prev_px <= 0):
                continue
            if di > 0 and px >= prev_px * 1.095:
                continue  # 涨停
            if di > 0 and px <= prev_px * 0.905:
                continue  # 跌停
            buyable.append(code)

        target_stocks = buyable[:buy_count]
        if not target_stocks:
            # 清仓
            for code in list(positions.keys()):
                pos = positions.pop(code)
                px = price_data.get(code, {}).get(ds)
                if px and px > 0:
                    sell_val = px * pos["shares"]
                    comm = max(sell_val * commission_rate, min_commission)
                    stamp = sell_val * stamp_rate
                    cash += sell_val - comm - stamp
                    trades.append(BacktestTrade(ds, code, name_map.get(code, ""),
                                                "sell", px, pos["shares"], sell_val,
                                                "清仓", 0, cash))
            continue

        # 卖出不在目标中的持仓
        for code in list(positions.keys()):
            if code not in target_stocks:
                pos = positions.pop(code)
                px = price_data.get(code, {}).get(ds)
                if px and px > 0:
                    sell_val = px * pos["shares"]
                    comm = max(sell_val * commission_rate, min_commission)
                    stamp = sell_val * stamp_rate
                    cash += sell_val - comm - stamp
                    trades.append(BacktestTrade(ds, code, name_map.get(code, ""),
                                                "sell", px, pos["shares"], sell_val,
                                                "调仓卖出", len(positions), cash))

        # 买入新目标
        n_pos = len(positions)
        if buy_count > n_pos:
            alloc = cash / (buy_count - n_pos)
        else:
            alloc = 0

        for code in target_stocks:
            if code in positions:
                continue
            if alloc <= 0 or cash <= 0:
                break
            px = price_data.get(code, {}).get(ds)
            if not px or px <= 0:
                continue
            shares = int(alloc / px / 100) * 100
            if shares <= 0:
                continue
            cost = shares * px + max(shares * px * commission_rate, min_commission)
            if cost > cash:
                shares = int((cash - min_commission) / px / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * px + max(shares * px * commission_rate, min_commission)
            cash -= cost
            positions[code] = {"shares": shares, "buy_price": px}
            trades.append(BacktestTrade(ds, code, name_map.get(code, ""),
                                        "buy", px, shares, shares * px,
                                        "调仓买入", len(positions), cash))

    # 计算绩效
    nav_df = pd.DataFrame(daily_nav).set_index("date")
    nav_series = nav_df["total_value"].astype(float)
    final_value = nav_series.iloc[-1] if len(nav_series) > 0 else init_cash

    total_return = (final_value / init_cash - 1) * 100
    years = len(nav_series) / 252
    annual_return = ((final_value / init_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

    peak = nav_series.cummax()
    dd = (nav_series - peak) / peak
    max_dd = dd.min() * 100

    daily_ret = nav_series.pct_change().dropna()
    excess = daily_ret - 0.02 / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

    # 月度收益
    monthly = {}
    nav_idx = pd.to_datetime(nav_series.index)
    nav_pd = pd.Series(nav_series.values, index=nav_idx)
    monthly_groups = nav_pd.resample("ME")
    for period, grp in monthly_groups:
        if len(grp) > 0:
            monthly[period.strftime("%Y-%m")] = (grp.iloc[-1] / grp.iloc[0] - 1) * 100

    return BacktestResult(
        strategy_name="小市值策略",
        start_date=trade_dates[0].strftime("%Y-%m-%d"),
        end_date=trade_dates[-1].strftime("%Y-%m-%d"),
        init_cash=init_cash,
        final_value=final_value,
        total_return_pct=total_return,
        annual_return_pct=annual_return,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        trade_count=len(trades),
        trading_days=len(trade_dates),
        nav_series=nav_series,
        drawdown_series=dd * 100,
        monthly_returns=monthly,
        trades=trades,
    )


def _backtest_generic(strategy, trade_dates, params, init_cash,
                      commission_rate, stamp_rate, min_commission,
                      rebalance_days=1):
    """通用回测（含调仓周期+等权分配+持仓上限）"""
    buy_count = int(params.get("max_hold", params.get("buy_count", 5)))
    cash = float(init_cash)
    positions = {}
    daily_nav = []
    trades = []

    for di, date in enumerate(trade_dates):
        ds = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]

        # 每日计算持仓市值
        pos_value = 0
        for code, pos in list(positions.items()):
            # 尝试获取当天价格（通过运行策略获取信号中的价格）
            px = pos.get("current_price", pos.get("buy_price", 0))
            pos_value += px * pos["shares"]

        total_value = cash + pos_value
        daily_nav.append({"date": ds, "total_value": total_value, "cash": cash})

        # 判断是否调仓日
        if di == 0:
            is_rebalance = True
            last_rebalance_date = date
        else:
            days_since = (date - last_rebalance_date).days
            is_rebalance = days_since >= rebalance_days
        if not is_rebalance:
            continue
        if di > 0:
            last_rebalance_date = date

        # 调仓日：运行策略（传入当前持仓，让策略自行判断买卖）
        current_hold_codes = list(positions.keys())
        result = strategy.run(params, date=_to_hikyuu_date(date), holdings=current_hold_codes or None)
        buy_codes = [s.code for s in result.signals if s.signal == "buy"]
        sell_codes = set(s.code for s in result.signals if s.signal == "sell")

        # 用策略信号中的价格更新持仓现价
        price_map = {s.code: s.price for s in result.signals if s.price > 0}

        # 重新计算持仓市值（用当天价格）
        pos_value = 0
        for code, pos in list(positions.items()):
            px = price_map.get(code, pos.get("buy_price", 0))
            pos["current_price"] = px
            pos_value += px * pos["shares"]
        total_value = cash + pos_value
        # 更新当天NAV
        daily_nav[-1] = {"date": ds, "total_value": total_value, "cash": cash}

        if di == 0:
            continue  # 第一天只记录

        # 卖出: 仅卖出策略明确标记 sell 的股票
        # 策略的 run() 已自行对比 holdings 和选出条件，只标记应卖出的
        sell_targets = set()
        if not buy_codes and positions:
            # 没有买入信号时，全部卖出（清仓）
            sell_targets = set(positions.keys())
        else:
            for code in list(positions.keys()):
                if code in sell_codes:
                    sell_targets.add(code)

        for code in sell_targets:
            if code not in positions:
                continue
            pos = positions.pop(code)
            px = price_map.get(code, pos.get("buy_price", 0))
            if px > 0:
                sell_val = px * pos["shares"]
                comm = max(sell_val * commission_rate, min_commission)
                stamp = sell_val * stamp_rate
                cash += sell_val - comm - stamp
                n_hold = len(positions)
                trades.append(BacktestTrade(ds, code,
                    result.signals[0].name if result.signals else code,
                    "sell", px, pos["shares"], sell_val,
                    "策略卖出" if code in sell_codes else "调出",
                    n_hold, cash))

        # 买入: 等权分配，不超过持仓上限
        slots = buy_count - len(positions)
        if slots > 0 and buy_codes and cash > 0:
            alloc = cash / slots
            for sig in result.signals:
                if sig.signal != "buy":
                    continue
                code = sig.code
                if code in positions:
                    continue
                price = sig.price
                if price <= 0:
                    continue
                shares = int(alloc / price / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * price + max(shares * price * commission_rate, min_commission)
                if cost > cash:
                    shares = int((cash - min_commission) / price / 100) * 100
                    if shares <= 0:
                        continue
                    cost = shares * price + max(shares * price * commission_rate, min_commission)
                cash -= cost
                positions[code] = {"shares": shares, "buy_price": price}
                trades.append(BacktestTrade(ds, code, sig.name, "buy",
                                            price, shares, shares * price,
                                            "策略买入", len(positions), cash))

    nav_df = pd.DataFrame(daily_nav).set_index("date")
    nav_series = nav_df["total_value"].astype(float)
    final_value = nav_series.iloc[-1] if len(nav_series) > 0 else init_cash
    total_return = (final_value / init_cash - 1) * 100
    years = len(nav_series) / 252
    annual_return = ((final_value / init_cash) ** (1 / years) - 1) * 100 if years > 0 else 0
    peak = nav_series.cummax()
    dd = (nav_series - peak) / peak
    max_dd = dd.min() * 100
    daily_ret = nav_series.pct_change().dropna()
    excess = daily_ret - 0.02 / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

    return BacktestResult(
        strategy_name=strategy.name,
        start_date=_to_date_str(trade_dates[0]),
        end_date=_to_date_str(trade_dates[-1]),
        init_cash=init_cash, final_value=final_value,
        total_return_pct=total_return, annual_return_pct=annual_return,
        max_drawdown_pct=max_dd, sharpe_ratio=sharpe,
        trade_count=len(trades), trading_days=len(trade_dates),
        nav_series=nav_series, drawdown_series=dd * 100,
        trades=trades,
    )


# ── ETF双池动量轮动 专用回测 ──

_ETF_PRICE_CACHE = {}
_ETF_VOL_CACHE = {}


def _get_etf_price(code, query_date):
    """获取 ETF 在 query_date 的收盘价（带缓存）"""
    from hikyuu import Datetime as HkuDt
    cache_key = f"{code}_{_to_date_str(query_date)}"
    if cache_key in _ETF_PRICE_CACHE:
        return _ETF_PRICE_CACHE[cache_key]
    try:
        if isinstance(query_date, datetime):
            hk_dt = _to_hikyuu_date(query_date)
        elif isinstance(query_date, HkuDt):
            hk_dt = query_date
        else:
            hk_dt = _to_hikyuu_date(str(query_date)[:10])
        q = Query(hk_dt, hk_dt + Days(1))
        stk = sm[code.lower()]
        k = stk.get_kdata(q)
        if k and len(k) > 0:
            px = float(k[-1].close)
        else:
            px = 0.0
        _ETF_PRICE_CACHE[cache_key] = px
        return px
    except Exception:
        return 0.0


def _get_etf_volume(code, query_date):
    """获取 ETF 在 query_date 的成交量（带缓存）"""
    cache_key = f"vol_{code}_{_to_date_str(query_date)}"
    if cache_key in _ETF_VOL_CACHE:
        return _ETF_VOL_CACHE[cache_key]
    try:
        hk_dt = _to_hikyuu_date(query_date) if not isinstance(query_date, Datetime) else query_date
        q = Query(hk_dt, hk_dt + Days(1))
        stk = sm[code.lower()]
        k = stk.get_kdata(q)
        if k and len(k) > 0:
            vol = float(k[-1].volume)
        else:
            vol = 0.0
        _ETF_VOL_CACHE[cache_key] = vol
        return vol
    except Exception:
        return 0.0


def _get_etf_avg_vol_5d(code, query_date):
    """获取 ETF 过去5日均量（不含 query_date 当日，无未来函数）"""
    try:
        hk_dt = _to_hikyuu_date(query_date) if not isinstance(query_date, Datetime) else query_date
        # 用自然日偏移，确保查到的数据以 query_date 为截止
        from datetime import timedelta
        if hasattr(hk_dt, 'datetime'):
            py_start = hk_dt.datetime() - timedelta(days=60)
        else:
            py_start = datetime(hk_dt.year, hk_dt.month, hk_dt.day) - timedelta(days=60)
        hk_start = _to_hikyuu_date(py_start)
        q = Query(hk_start, hk_dt)
        stk = sm[code.lower()]
        k = stk.get_kdata(q)
        if not k or len(k) < 6:
            return 0.0
        vols = [float(r.volume) for r in k]
        recent = [v for v in vols if v > 0]
        return np.mean(recent[-6:-1]) if len(recent) >= 6 else 0.0
    except Exception:
        return 0.0


def _backtest_etf_momentum(trade_dates, params, init_cash,
                           commission_rate, stamp_rate, min_commission,
                           rebalance_days=1):
    """
    ETF双池平滑动量轮动 专用回测

    特性：
      - 每日止损检查（-8%）
      - 每日放量监控（>2.5x 5日均量）
      - 防御ETF处理（SH511880 跳过风控）
      - 调仓日运行策略生成信号
    """
    from trade_advisor.etf_pool import DEFENSIVE_ETF_CODE, get_etf_name

    buy_count = int(params.get("buy_count", 1))
    stop_loss_pct = float(params.get("stop_loss", 8)) / 100.0
    vol_ratio = float(params.get("volume_ratio", 2.5))

    strategy = REGISTERED_STRATEGIES["etf_momentum"]

    cash = float(init_cash)
    positions = {}          # code -> {"shares": int, "buy_price": float, "buy_value": float}
    daily_nav = []
    trades = []

    last_rebalance_date = trade_dates[0]

    for di, date in enumerate(trade_dates):
        ds = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]
        hk_date = _to_hikyuu_date(date)

        # ── A. 计算持仓市值 ──
        pos_value = 0
        for code, pos in list(positions.items()):
            px = _get_etf_price(code, hk_date)
            if px > 0:
                pos["current_price"] = px
                pos_value += px * pos["shares"]
            else:
                # 当天无数据，用买入价近似
                pos_value += pos.get("buy_price", 0) * pos["shares"]

        total_value = cash + pos_value
        daily_nav.append({"date": ds, "total_value": total_value, "cash": cash})

        # 第一天只记录不交易
        if di == 0:
            continue

        # ── B. 每日风控检查（止损 + 放量） ──
        risk_sells = []
        for code, pos in list(positions.items()):
            # 防御ETF跳过风控
            if code == DEFENSIVE_ETF_CODE:
                continue

            current_price = _get_etf_price(code, hk_date)
            if current_price <= 0:
                continue

            # B1. 止损
            cost = pos["buy_price"]
            if cost > 0 and current_price < cost * (1 - stop_loss_pct):
                risk_sells.append((code, f"止损(现价{current_price:.3f}<成本{cost:.3f})"))
                continue

            # B2. 放量排雷
            current_vol = _get_etf_volume(code, hk_date)
            avg_vol_5d = _get_etf_avg_vol_5d(code, hk_date)
            if avg_vol_5d > 0 and current_vol > avg_vol_5d * vol_ratio:
                ratio = current_vol / avg_vol_5d
                risk_sells.append((code, f"放量({ratio:.1f}x 5日均量)"))

        # 执行风控卖出
        for code, reason in risk_sells:
            if code not in positions:
                continue
            pos = positions.pop(code)
            px = pos.get("current_price", pos.get("buy_price", 0))
            if px > 0:
                sell_val = px * pos["shares"]
                comm = max(sell_val * commission_rate, min_commission)
                stamp = sell_val * stamp_rate
                cash += sell_val - comm - stamp
                trades.append(BacktestTrade(
                    ds, code, get_etf_name(code),
                    "sell", px, pos["shares"], sell_val,
                    reason, len(positions), cash,
                ))

        # ── C. 判断调仓日 ──
        days_since = (date - last_rebalance_date).days
        if days_since < rebalance_days:
            continue
        last_rebalance_date = date

        # ── D. 调仓日：运行策略 ──
        current_hold_codes = list(positions.keys())
        result = strategy.run(params, date=hk_date, holdings=current_hold_codes or None)
        buy_codes = [s.code for s in result.signals if s.signal == "buy"]
        sell_codes = set(s.code for s in result.signals if s.signal == "sell")
        price_map = {s.code: s.price for s in result.signals if s.price > 0}

        # 更新持仓现价
        for code, pos in positions.items():
            px = price_map.get(code, pos.get("current_price", pos.get("buy_price", 0)))
            pos["current_price"] = px

        # D1. 卖出
        if not buy_codes and positions:
            # 无买入信号 → 清仓
            sell_targets = set(positions.keys())
        else:
            sell_targets = set()
            for code in list(positions.keys()):
                if code in sell_codes:
                    sell_targets.add(code)

        for code in sell_targets:
            if code not in positions:
                continue
            pos = positions.pop(code)
            px = price_map.get(code, pos.get("current_price", pos.get("buy_price", 0)))
            if px > 0:
                sell_val = px * pos["shares"]
                comm = max(sell_val * commission_rate, min_commission)
                stamp = sell_val * stamp_rate
                cash += sell_val - comm - stamp
                trades.append(BacktestTrade(
                    ds, code, get_etf_name(code),
                    "sell", px, pos["shares"], sell_val,
                    "调仓卖出", len(positions), cash,
                ))

        # D2. 买入
        slots = buy_count - len(positions)
        if slots > 0 and buy_codes and cash > 0:
            alloc = cash / slots
            for sig in result.signals:
                if sig.signal != "buy":
                    continue
                code = sig.code
                if code in positions:
                    continue
                price = sig.price
                if price <= 0:
                    continue
                shares = int(alloc / price / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * price + max(shares * price * commission_rate, min_commission)
                if cost > cash:
                    shares = int((cash - min_commission) / price / 100) * 100
                    if shares <= 0:
                        continue
                    cost = shares * price + max(shares * price * commission_rate, min_commission)
                cash -= cost
                positions[code] = {"shares": shares, "buy_price": price, "current_price": price}
                trades.append(BacktestTrade(
                    ds, code, sig.name, "buy",
                    price, shares, shares * price,
                    "策略买入", len(positions), cash,
                ))

    # ── E. 计算绩效 ──
    nav_df = pd.DataFrame(daily_nav).set_index("date")
    nav_series = nav_df["total_value"].astype(float)
    final_value = nav_series.iloc[-1] if len(nav_series) > 0 else init_cash
    total_return = (final_value / init_cash - 1) * 100
    years = len(nav_series) / 252
    annual_return = ((final_value / init_cash) ** (1 / years) - 1) * 100 if years > 0 else 0
    peak = nav_series.cummax()
    dd = (nav_series - peak) / peak
    max_dd = dd.min() * 100
    daily_ret = nav_series.pct_change().dropna()
    excess = daily_ret - 0.02 / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

    # 月度收益
    monthly = {}
    try:
        monthly_df = nav_df.resample("ME")
        if len(monthly_df) > 1:
            monthly_vals = monthly_df["total_value"].last()
            monthly_ret = monthly_vals.pct_change().dropna() * 100
            for m, r in monthly_ret.items():
                monthly[str(m)[:7]] = round(r, 2)
    except Exception:
        pass

    return BacktestResult(
        strategy_name="ETF双池动量轮动",
        start_date=_to_date_str(trade_dates[0]),
        end_date=_to_date_str(trade_dates[-1]),
        init_cash=init_cash, final_value=final_value,
        total_return_pct=total_return, annual_return_pct=annual_return,
        max_drawdown_pct=max_dd, sharpe_ratio=sharpe,
        trade_count=len(trades), trading_days=len(trade_dates),
        nav_series=nav_series, drawdown_series=dd * 100,
        trades=trades, monthly_returns=monthly,
    )


