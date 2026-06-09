"""
Hikyuu 回测：中小盘小市值策略
原策略来自聚宽：https://www.joinquant.com/post/25496

策略逻辑：
  1. 股票池：中小板指 (399101.XSHE) 全部成分股
  2. 每天选流通市值最小的5只股票
  3. 过滤：ST、涨停、跌停、停牌
  4. 等资金分配买入，持仓不在买入列表则卖出
  5. 回测周期：最近一年

消除未来函数：
  - 选股基于「前一交易日」收盘数据（流通市值、排序）
  - 交易在「当日」收盘执行（买入/卖出价）
  - 涨跌停/停牌检查使用当日数据（收盘时已可知）
  - 模拟真实场景：T-1 日收盘决策，T 日收盘执行

用法：
  conda activate stock
  python backtest_small_cap.py
"""

from hikyuu.interactive import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_stock_name_safe(stock):
    """获取股票名称，处理可能的编码问题"""
    try:
        return stock.name.encode("latin-1").decode("gbk", errors="replace")
    except Exception:
        return stock.name


def is_st_stock(stock):
    """判断是否为ST/*ST/退市股票"""
    name = get_stock_name_safe(stock)
    return "ST" in name or "*" in name or "退" in name


def build_price_matrix(stocks, query):
    """
    预加载所有股票的日线数据，返回 (date_index, price_df, vol_df)
    date_index: 共同的交易日列表
    price_df: 日期 x 股票代码 的收盘价矩阵
    vol_df:   日期 x 股票代码 的成交量矩阵
    """
    print("正在加载K线数据...")
    all_close = {}
    all_vol = {}
    all_high = {}
    all_low = {}
    total = len(stocks)
    valid_stocks = []

    for i, stock in enumerate(stocks):
        if i % 100 == 0 and i > 0:
            print(f"  进度: {i}/{total}")
        try:
            k = stock.get_kdata(query)
            if not k or len(k) == 0:
                continue

            dates = [r.datetime.datetime() for r in k]
            closes = [float(r.close) for r in k]
            volumes = [float(r.volume) for r in k]
            highs = [float(r.high) for r in k]
            lows = [float(r.low) for r in k]

            code = stock.market_code
            all_close[code] = pd.Series(closes, index=dates, dtype=np.float64)
            all_vol[code] = pd.Series(volumes, index=dates, dtype=np.float64)
            all_high[code] = pd.Series(highs, index=dates, dtype=np.float64)
            all_low[code] = pd.Series(lows, index=dates, dtype=np.float64)
            valid_stocks.append((stock, code))
        except Exception as e:
            print(f"  警告: {stock.market_code} 数据加载失败: {e}")

    print(f"  进度: {total}/{total} — 有效股票: {len(valid_stocks)}")

    if not valid_stocks:
        raise RuntimeError("没有有效股票数据")

    # 合并为DataFrame，缺失值用NaN填充
    price_df = pd.DataFrame(all_close)
    vol_df = pd.DataFrame(all_vol)
    high_df = pd.DataFrame(all_high)
    low_df = pd.DataFrame(all_low)

    # 删除全空列
    price_df = price_df.dropna(axis=1, how="all")
    vol_df = vol_df.dropna(axis=1, how="all")
    high_df = high_df.dropna(axis=1, how="all")
    low_df = low_df.dropna(axis=1, how="all")

    # 只保留共同的索引
    common_idx = price_df.index.intersection(vol_df.index)
    price_df = price_df.loc[common_idx].sort_index()
    vol_df = vol_df.loc[common_idx].sort_index()
    high_df = high_df.loc[common_idx].sort_index()
    low_df = low_df.loc[common_idx].sort_index()

    return valid_stocks, price_df, vol_df, high_df, low_df


def filter_limitup_stock(price_row, high_row, prev_close_row, code, held_positions):
    """过滤涨停股"""
    # 已持仓的不过滤
    if code in held_positions:
        return True
    if pd.isna(price_row.get(code)) or pd.isna(prev_close_row.get(code)):
        return False
    limit_up = prev_close_row[code] * 1.095  # 留一点容差
    return high_row[code] < limit_up


def filter_limitdown_stock(low_row, prev_close_row, code):
    """过滤跌停股"""
    if pd.isna(low_row.get(code)) or pd.isna(prev_close_row.get(code)):
        return False
    limit_down = prev_close_row[code] * 0.905  # 留一点容差
    return low_row[code] > limit_down


def filter_paused_stock(vol_row, code):
    """过滤停牌股（成交量为0）"""
    if pd.isna(vol_row.get(code)):
        return False
    return vol_row[code] > 0


def backtest_small_cap_strategy(
    lookback_days=365,
    buy_stock_count=5,
    initial_cash=100_000,
    commission_rate=0.0003,
    stamp_rate=0.001,
    min_commission=5.0,
):
    """
    中小盘小市值策略回测

    参数:
        lookback_days: 回测天数
        buy_stock_count: 持仓股票数量
        initial_cash: 初始资金
        commission_rate: 佣金费率
        stamp_rate: 印花税费率
        min_commission: 最低佣金
    """
    # ── 1. 获取股票池 ──
    print("=" * 60)
    print("中小盘小市值策略回测")
    print("=" * 60)

    idx_stock = sm["sz399101"]
    blocks = sm.get_block_list_by_index_stock(idx_stock)
    idx_block = blocks[0]
    all_stocks = idx_block.get_stock_list()
    print(f"中小板指成分股: {len(all_stocks)} 只")

    # ── 2. 确定回测区间 ──
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    start_dt = Datetime(start.year, start.month, start.day)
    end_dt = Datetime(end.year, end.month, end.day)
    print(f"回测区间: {start.date()} ~ {end.date()}")

    # 查询多取一些数据（用于前值计算）
    query_start = start - timedelta(days=30)
    q = Query(
        Datetime(query_start.year, query_start.month, query_start.day),
        end_dt,
    )

    # ── 3. 预加载数据 ──
    valid_stocks, price_df, vol_df, high_df, low_df = build_price_matrix(all_stocks, q)
    stock_map = {code: stock for stock, code in valid_stocks}

    # 过滤掉ST股票
    non_st_codes = []
    for stock, code in valid_stocks:
        if not is_st_stock(stock):
            non_st_codes.append(code)
    print(f"非ST股票: {len(non_st_codes)} 只")

    # ── 4. 获取交易日历 ──
    # 只处理所有股票都有数据的交易日
    trade_dates = price_df.index.sort_values()
    trade_dates = trade_dates[
        (trade_dates >= start) & (trade_dates <= end)
    ]
    print(f"交易日数: {len(trade_dates)}")

    if len(trade_dates) < 20:
        raise RuntimeError("交易日太少，请检查数据是否完整")

    # ── 5. 预加载财务数据（流通股本） ──
    print("\n正在加载财务数据（流通股本）...")
    liutongguben = {}  # code -> float
    for stock, code in valid_stocks:
        try:
            fin = stock.get_finance_info()
            liutongguben[code] = float(fin["liutongguben"])
        except Exception:
            pass
    print(f"  流通股本数据: {len(liutongguben)} 只")

    # ── 6. 主回测循环 ──
    print("\n开始回测...")
    cash = float(initial_cash)
    positions = {}  # code -> {shares, cost_basis}
    daily_values = []

    # 交易统计
    trade_records = []
    total_trades = 0

    for di, date in enumerate(trade_dates):
        if di % 50 == 0 and di > 0:
            print(f"  进度: {di}/{len(trade_dates)}")

        # 获取当天及前一天数据
        price_row = price_df.loc[date]
        vol_row = vol_df.loc[date]
        high_row = high_df.loc[date]
        low_row = low_df.loc[date]

        # 前一天收盘价（用于计算涨跌停）
        if di == 0:
            # 第一天没有前值 —— 用当天开盘价近似（或直接跳过交易）
            prev_close_row = price_row
        else:
            prev_date = trade_dates[di - 1]
            prev_close_row = price_df.loc[prev_date]

        # 计算持仓市值
        position_value = 0
        for code, pos in list(positions.items()):
            if code in price_row and not pd.isna(price_row[code]):
                pos["current_price"] = float(price_row[code])
                position_value += pos["current_price"] * pos["shares"]
            else:
                # 股票停牌或无数据，使用前一日价格
                position_value += pos.get("cost_basis", 0) * pos["shares"]

        total_value = cash + position_value
        daily_values.append({"date": date, "total_value": total_value, "cash": cash})

        # 第一天只记录不交易（没有前收盘价用于判断涨跌停）
        if di == 0:
            continue

        # ── 6a. 选股（用前一交易日数据，消除未来函数） ──
        # 核心原则：回测只能使用「交易时刻已存在的数据」
        # 在 T 日收盘时，我们能看到的只有 T-1 日的收盘数据
        # 因此用 T-1 收盘价计算流通市值进行选股，T 日收盘执行交易
        prev_date = trade_dates[di - 1]
        prev_price_row = price_df.loc[prev_date]

        # 计算流通市值 = 流通股本 × 前收盘价
        caps = {}
        for code in non_st_codes:
            if code in prev_price_row and code in liutongguben and not pd.isna(prev_price_row[code]):
                cap = liutongguben[code] * prev_price_row[code]
                if cap > 0:
                    caps[code] = cap

        # 按流通市值升序排列，取最多的候选（3倍目标数）
        sorted_caps = sorted(caps.items(), key=lambda x: x[1])
        candidates = [code for code, _ in sorted_caps[: buy_stock_count * 3]]

        # 过滤涨停、跌停、停牌
        buyable = []
        for code in candidates:
            if not filter_limitup_stock(
                price_row, high_row, prev_close_row, code, positions
            ):
                continue
            if not filter_limitdown_stock(low_row, prev_close_row, code):
                continue
            if not filter_paused_stock(vol_row, code):
                continue
            buyable.append(code)

        # 取最终要买入的只数
        target_stocks = buyable[:buy_stock_count]
        # 如果没有可选股票，清仓等待
        if not target_stocks:
            # 清仓所有持仓
            for code in list(positions.keys()):
                pos = positions.pop(code)
                if code in price_row and not pd.isna(price_row[code]):
                    sell_price = float(price_row[code])
                    sell_value = sell_price * pos["shares"]
                    commission = max(sell_value * commission_rate, min_commission)
                    stamp = sell_value * stamp_rate
                    cash += sell_value - commission - stamp
                    trade_records.append(
                        {
                            "date": date,
                            "code": code,
                            "action": "SELL",
                            "price": sell_price,
                            "shares": pos["shares"],
                            "value": sell_value,
                            "reason": "清仓-无目标",
                        }
                    )
                    total_trades += 1
            continue

        # ── 6b. 卖出不在目标列表中的持仓 ──
        for code in list(positions.keys()):
            if code not in target_stocks:
                pos = positions.pop(code)
                if code in price_row and not pd.isna(price_row[code]):
                    sell_price = float(price_row[code])
                    sell_value = sell_price * pos["shares"]
                    commission = max(sell_value * commission_rate, min_commission)
                    stamp = sell_value * stamp_rate
                    cash += sell_value - commission - stamp
                    trade_records.append(
                        {
                            "date": date,
                            "code": code,
                            "action": "SELL",
                            "price": sell_price,
                            "shares": pos["shares"],
                            "value": sell_value,
                            "reason": "调仓卖出",
                        }
                    )
                    total_trades += 1

        # ── 6c. 买入新目标 ──
        # 计算每只股票可用资金
        position_count = len(positions)
        if buy_stock_count > position_count:
            alloc_cash = cash / (buy_stock_count - position_count)
        else:
            alloc_cash = 0

        for code in target_stocks:
            if code in positions:
                continue  # 已持有
            if alloc_cash <= 0 or cash <= 0:
                break

            if code not in price_row or pd.isna(price_row[code]):
                continue

            buy_price = float(price_row[code])
            # 计算可买数量（按100股整数倍）
            shares = int(alloc_cash / buy_price / 100) * 100
            if shares <= 0:
                continue

            commission = max(shares * buy_price * commission_rate, min_commission)
            cost = shares * buy_price + commission
            if cost > cash:
                # 重新计算：按实际现金买
                shares = int((cash - min_commission) / buy_price / 100) * 100
                if shares <= 0:
                    continue
                commission = max(shares * buy_price * commission_rate, min_commission)
                cost = shares * buy_price + commission

            cash -= cost
            positions[code] = {
                "shares": shares,
                "cost_basis": buy_price,
                "current_price": buy_price,
            }
            trade_records.append(
                {
                    "date": date,
                    "code": code,
                    "action": "BUY",
                    "price": buy_price,
                    "shares": shares,
                    "value": shares * buy_price,
                    "reason": "买入",
                }
            )
            total_trades += 1

    # ── 7. 计算结果 ──
    print("\n回测完成，计算绩效...")

    # 计算每日净值序列
    nav_df = pd.DataFrame(daily_values).set_index("date")
    nav_df["daily_return"] = nav_df["total_value"].pct_change()

    # 最终结果
    final_value = nav_df["total_value"].iloc[-1] if len(nav_df) > 0 else initial_cash
    total_return = (final_value / initial_cash - 1) * 100

    # 年化收益率
    years = len(nav_df) / 252  # 交易日数 / 252
    annual_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 最大回撤
    peak = nav_df["total_value"].cummax()
    drawdown = (nav_df["total_value"] - peak) / peak
    max_drawdown = drawdown.min() * 100

    # 夏普比（无风险利率按2%算）
    excess_returns = nav_df["daily_return"].dropna() - 0.02 / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

    # ── 8. 输出 ──
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"  初始资金:        {initial_cash:>12,.2f}")
    print(f"  最终总资产:      {final_value:>12,.2f}")
    print(f"  总收益率:        {total_return:>+11.2f}%")
    print(f"  年化收益率:      {annual_return:>+11.2f}%")
    print(f"  最大回撤:        {max_drawdown:>+11.2f}%")
    print(f"  夏普比率:        {sharpe:>11.4f}")
    print(f"  总交易次数:      {total_trades:>11}")
    print(f"  回测天数:        {len(nav_df):>11}")

    # 月度收益
    nav_df["month"] = nav_df.index.to_period("M")
    monthly_ret = nav_df.groupby("month")["total_value"].last().pct_change() * 100
    print("\n  月度收益率:")
    for m, r in monthly_ret.dropna().items():
        sign = "+" if r >= 0 else ""
        print(f"    {m}: {sign}{r:.2f}%")

    # 最近5笔交易
    print(f"\n  最近5笔交易:")
    for t in trade_records[-5:]:
        sign = "+" if t["action"] == "BUY" else "-"
        print(
            f"    {t['date'].strftime('%Y-%m-%d')} | {sign} {t['action']:4} | "
            f"{t['code']} | {t['price']:>8.2f} | {t['shares']:>5}股 | {t['value']:>10.2f}"
        )

    print("\n" + "=" * 60)

    # ── 9. 画图 ──
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rc("font", family="Microsoft YaHei", size=10)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # 资产曲线
        ax1 = axes[0]
        plot_dates = nav_df.index if isinstance(nav_df.index, pd.DatetimeIndex) else pd.DatetimeIndex(nav_df.index)
        ax1.plot(
            plot_dates,
            nav_df["total_value"],
            label="总资产",
            color="steelblue",
            linewidth=1.5,
        )
        ax1.axhline(y=initial_cash, color="gray", linestyle="--", alpha=0.5, label="初始本金")
        ax1.set_ylabel("总资产 (元)")
        ax1.set_title("中小盘小市值策略回测 — 资产曲线")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 回撤曲线
        ax2 = axes[1]
        dd_dates = drawdown.index if isinstance(drawdown.index, pd.DatetimeIndex) else pd.DatetimeIndex(drawdown.index)
        ax2.fill_between(
            dd_dates,
            drawdown.values * 100,
            0,
            color="crimson",
            alpha=0.3,
            label="回撤",
        )
        ax2.set_ylabel("回撤 (%)")
        ax2.set_xlabel("日期")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("backtest_small_cap_result.png", dpi=150, bbox_inches="tight")
        print("  图表已保存: backtest_small_cap_result.png")
        plt.show()
    except ImportError:
        print("  [跳过] matplotlib 未安装，跳过画图")
    except Exception as e:
        print(f"  [跳过] 画图出错: {e}")

    return nav_df, trade_records, drawdown


if __name__ == "__main__":
    backtest_small_cap_strategy(
        lookback_days=365,
        buy_stock_count=5,
        initial_cash=100_000,
    )
