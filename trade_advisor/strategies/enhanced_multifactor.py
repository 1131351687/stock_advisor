"""
增强版多因子策略 — 迁移自聚宽 ML 多因子模型选股

聚宽原文: https://www.joinquant.com/post/74040
作者: papacai0606.eth

核心机制（三级防御体系）：
  1️⃣ 市场风险监测 — 市场宽度、行业集中度、Z-score
  2️⃣ 多因子评分 — 趋势/资金/筹码/基本面四维评分
  3️⃣ 风险自适应 — 高风险切ETF防御，中风险降仓位

与原始策略的对应关系：
  聚宽                          → 本实现
  ─────────────────────────────────────────
  MarketRiskMonitor             → MarketRiskMonitor (Hikyuu版)
  LightGBM ML模型                → 因子加权评分（可扩展为ML）
  get_fundamentals + ROE筛选    → Hikyuu财务数据
  run_weekly + weekly_adjustment → 回测引擎调仓周期
  ETF防御池                      → etf_pool 防御ETF
  check_risk 止盈止损            → risk_adjusted_stop

数据源：Hikyuu（全本地）
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

from hikyuu.interactive import *
from trade_advisor.strategies import (
    BaseStrategy, StrategyParam, StockSignal, StrategyResult,
    _to_hikyuu_date, _to_date_str, _get_last_trade_date,
    get_stock_display_name, is_st_stock,
)
from trade_advisor.strategies.market_risk_monitor import MarketRiskMonitor
from trade_advisor.data_adapter import get_kdata, get_stock_list, get_market_cap


# ── 因子权重体系 ──
# 总分为 100，分配如下：
FACTOR_WEIGHTS = {
    "trend":    25,   # 趋势因子 — 均线排列 + 价格位置
    "momentum": 25,   # 动量因子 — 近期涨幅 + 新高
    "volume":   20,   # 量能因子 — 量比 + 资金介入
    "quality":  15,   # 质量因子 — ROE + 毛利率(财务)
    "turnover": 15,   # 筹码因子 — 换手率
}


class EnhancedMultiFactorStrategy(BaseStrategy):
    """
    增强版多因子策略 — 市场风险监测 + 四维评分 + 风险自适应

    三级防御：
      risk_level=0: 正常选股，满分仓
      risk_level=1: 降至80%仓位，收紧止损
      risk_level=2: 切换至防御ETF，严格止损
    """

    def __init__(self):
        super().__init__()
        self.risk_monitor = MarketRiskMonitor()

    @property
    def name(self) -> str:
        return "增强多因子"

    @property
    def description(self) -> str:
        return "市场风险监测+四维因子评分+风险自适应(高风控切ETF)"

    @property
    def params(self) -> list:
        return [
            StrategyParam("buy_count",    "持仓数量",   5,  "int", min_val=1,   max_val=20),
            StrategyParam("score_threshold","买入阈值", 30, "int", min_val=10,  max_val=90),
            StrategyParam("trend_weight", "趋势因子权重",25, "int", min_val=0,  max_val=40),
            StrategyParam("momentum_weight","动量因子权重",25,"int", min_val=0, max_val=40),
            StrategyParam("volume_weight","量能因子权重",20, "int", min_val=0,  max_val=40),
            StrategyParam("stop_loss",    "止损比例(%)", 5, "float",min_val=2,  max_val=15, step=1),
            StrategyParam("risk_mode",    "风控模式",
                          "auto", "choice", options=["auto", "always_stock", "always_etf"]),
            StrategyParam("etf_fallback", "高风险切ETF", 1, "int", min_val=0, max_val=1),
        ]

    def _score_trend(self, closes: np.ndarray) -> float:
        """趋势因子评分 (0-25)"""
        if len(closes) < 25:
            return 0

        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        ma60 = np.mean(closes[-60:]) if len(closes) >= 60 else ma20
        c = closes[-1]

        score = 0
        # 多头排列 - 逐层累加
        if ma10 > ma20:
            score += 8  # 中期趋势向上
        if ma5 > ma10:
            score += 7  # 短期趋势更强

        # 价格位置
        above_ma20 = (c - ma20) / ma20 * 100
        if above_ma20 > 3:
            score += 5  # 明显在MA20上方
        elif above_ma20 > 0:
            score += 3  # 刚站上MA20

        # MA60 方向
        if len(closes) >= 65:
            ma60_now = np.mean(closes[-5:])
            ma60_prev = np.mean(closes[-10:-5])
            if ma60_now > ma60_prev:
                score += 5  # 长期趋势向上

        return min(25, score)

    def _score_momentum(self, closes: np.ndarray, highs: np.ndarray) -> float:
        """动量因子评分 (0-25)"""
        if len(closes) < 25:
            return 0

        c = closes[-1]
        score = 0

        # 20日涨幅评分 (线性)
        ret_20d = c / closes[-20] - 1
        if ret_20d > 0:
            score += min(15, ret_20d * 150)  # 5%涨幅→7.5分, 10%→15分

        # 接近/突破20日新高
        high_20 = np.max(highs[-20:])
        if c >= high_20:
            score += 10
        elif high_20 > 0:
            ratio = c / high_20
            if ratio > 0.95:
                score += (ratio - 0.95) * 200  # 95%→0分, 100%→10分

        return min(25, score)

    def _score_volume(self, volumes: np.ndarray) -> float:
        """量能因子评分 (0-20)"""
        if len(volumes) < 6:
            return 0

        v = volumes[-1]
        vol_ma5 = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else np.mean(volumes)
        vol_ratio = v / vol_ma5 if vol_ma5 > 0 else 0

        if vol_ratio > 2.0:
            return 20
        elif vol_ratio > 1.0:
            return 10 + (vol_ratio - 1.0) * 10  # 1.0→10分, 2.0→20分
        elif vol_ratio > 0.7:
            return vol_ratio * 10  # 0.7→7分
        return vol_ratio * 5  # 0.5→2.5分

    def _score_quality(self, stk) -> float:
        """质量因子评分 (0-15) — 从 Hikyuu 财务数据获取"""
        try:
            fin = stk.get_finance_info()
            if not fin:
                return 5  # 默认中位分

            score = 0

            # ROE = 净利润 / 净资产 (>15% +5)
            if fin.have('jinglirun') and fin.have('jingzichan') and float(fin['jingzichan']) > 0:
                roe = float(fin['jinglirun']) / float(fin['jingzichan'])
                if roe > 0.15:
                    score += 5
                elif roe > 0.05:
                    score += 3

            # 每股净资产 (>10 +5)
            if fin.have('meigujingzichan'):
                nav_ps = float(fin['meigujingzichan'])
                if nav_ps > 10:
                    score += 5
                elif nav_ps > 0:
                    score += 3

            # 长期负债率 = 长期负债 / 总资产 (适中 +5)
            if fin.have('changqifuzhai') and fin.have('zongzichan') and float(fin['zongzichan']) > 0:
                lr = float(fin['changqifuzhai']) / float(fin['zongzichan'])
                if lr < 0.3:
                    score += 5
                elif lr < 0.6:
                    score += 2

            return min(15, score)

        except:
            return 0

    def _score_turnover(self, vol: float, liutongguben: float) -> float:
        """筹码因子评分 (0-15) — 基于换手率"""
        if liutongguben <= 0:
            return 0

        # 换手率(%) = 成交量(股) / 流通股本(股) * 100
        turnover_pct = vol / liutongguben * 100 if liutongguben > 0 else 0

        if 3 <= turnover_pct <= 12:
            return 15
        elif 1 <= turnover_pct < 3:
            return 8
        elif 12 < turnover_pct <= 20:
            return 5
        return 0

    def _get_defensive_etf_signals(self, target_dt, date_str) -> StrategyResult:
        """切换到防御模式 — 选择防御 ETF"""
        from trade_advisor.etf_pool import DEFENSIVE_ETF_CODE

        defensive_codes = ['sh511880', 'sh511010']  # 货币ETF + 国债ETF

        signals = []
        for code in defensive_codes:
            try:
                stk = sm[code]
                k = stk.get_kdata(Query(target_dt, target_dt + Days(1)))
                close = float(k[-1].close) if k and len(k) > 0 else 0
                if close > 0:
                    signals.append(StockSignal(
                        code=code,
                        name=get_stock_display_name(stk),
                        signal='buy',
                        reason='防御模式-高风险',
                        price=close,
                        weight=0.5,
                    ))
            except:
                continue

        return StrategyResult(
            date=date_str,
            strategy_name=self.name,
            signals=signals,
            metadata={"risk_mode": True, "defensive": True},
        )

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        buy_count = int(param_values.get("buy_count", 5))
        score_threshold = int(param_values.get("score_threshold", 60))
        stop_loss_pct = float(param_values.get("stop_loss", 5))
        risk_mode = param_values.get("risk_mode", "auto")
        etf_fallback = int(param_values.get("etf_fallback", 1))

        # ── 日期 ──
        if date is not None:
            target_dt = _to_hikyuu_date(date)
        else:
            _, ds = _get_last_trade_date()
            if ds == "N/A":
                return StrategyResult(date="N/A", strategy_name=self.name)
            target_dt = _to_hikyuu_date(ds)
        date_str = _to_date_str(target_dt)

        if not sm.get_trading_calendar(Query(target_dt, target_dt + Days(1))):
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  metadata={"error": "当日非交易日"})

        # 数据存在性检查
        _ref = sm["sh510050"].get_kdata(Query(target_dt, target_dt + Days(1)))
        if not _ref or len(_ref) == 0:
            _, fb = _get_last_trade_date()
            if fb and fb != "N/A":
                target_dt = _to_hikyuu_date(fb)
                date_str = fb

        # ── 风险监测 ──
        risk_data = self.risk_monitor.calculate(sm, target_dt)
        risk_level = risk_data['risk_level']

        # 高风险 → 切防御
        if risk_level >= 2 and etf_fallback and risk_mode != "always_stock":
            result = self._get_defensive_etf_signals(target_dt, date_str)
            result.metadata.update({
                "risk_level": risk_level,
                "risk_summary": self.risk_monitor.summary(target_dt),
                "risk_mode": True,
            })
            return result

        # ── 股票池（取中小板指成分 + 全市场A股交集中流通性好的） ──
        stock_codes = []
        try:
            # 优先用中小板指
            idx = sm['sz399101']
            blk = sm.get_block_list_by_index_stock(idx)
            stock_codes = [s.market_code for s in blk]
        except:
            pass

        if len(stock_codes) < 50:
            stock_codes = get_stock_list('all')[:1000]

        # ── 扫描评分 ──
        query_start = (target_dt - Days(100)).datetime().date()
        query_end = target_dt.datetime().date()
        candidates = []

        def score_stock(code: str) -> Optional[Dict]:
            try:
                df = get_kdata(code, str(query_start), str(query_end))
                if df.empty or len(df) < 30:
                    return None

                closes = df['close'].values
                highs = df['high'].values
                volumes = df['volume'].values
                c = closes[-1]
                v = volumes[-1]

                stk = sm[code.lower()]
                name = get_stock_display_name(stk)
                if is_st_stock(stk):
                    return None

                # 获取财务数据
                fin = stk.get_finance_info()
                liutongguben = 0.0
                if fin and fin.have('liutongguben'):
                    liutongguben = float(fin['liutongguben'])

                # 流通市值过滤（最小5亿，最多500亿）
                if liutongguben > 0:
                    mv = liutongguben * c / 1e8  # 亿元
                    if mv > 500:
                        return None

                # 四维评分
                s_trend = self._score_trend(closes)
                s_momentum = self._score_momentum(closes, highs)
                s_volume = self._score_volume(volumes)
                s_quality = self._score_quality(stk)
                s_turnover = self._score_turnover(v, liutongguben)

                total = s_trend + s_momentum + s_volume + s_quality + s_turnover

                if total < score_threshold:
                    return None

                return {
                    "code": code,
                    "name": name,
                    "score": total,
                    "price": c,
                    "factors": {
                        "trend": s_trend,
                        "momentum": s_momentum,
                        "volume": s_volume,
                        "quality": s_quality,
                        "turnover": s_turnover,
                    },
                }
            except:
                return None

        # 顺序扫描（Hikyuu C++ 后端非线程安全）
        max_scan = min(800, len(stock_codes))
        for i, code in enumerate(stock_codes[:max_scan]):
            if i % 100 == 0 and i > 0:
                print(f"  扫描进度: {i}/{max_scan}")
            r = score_stock(code)
            if r:
                candidates.append(r)

        if not candidates:
            # 无候选时尝试防御ETF
            if risk_level >= 1 and etf_fallback:
                return self._get_defensive_etf_signals(target_dt, date_str)
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  signals=[],
                                  metadata={"error": f"无股票评分>={score_threshold}",
                                            "scanned": len(stock_codes)})

        # ── 排序选股 ──
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:buy_count]

        # ── 仓位调整（基于风险等级） ──
        if risk_level >= 1:
            position_ratio = 0.8  # 中等风险 → 80%
        else:
            position_ratio = 1.0

        # ── 生成信号 ──
        signals = []
        buy_codes = set()
        weights = [1.0 / len(selected) * position_ratio] * len(selected)

        for i, item in enumerate(selected):
            code = item["code"]
            buy_codes.add(code)
            f = item["factors"]
            detail = f"趋势:{f['trend']}+动量:{f['momentum']}+量能:{f['volume']}"
            if f['quality'] > 0:
                detail += f"+质量:{f['quality']}"
            if f['turnover'] > 0:
                detail += f"+换手:{f['turnover']}"

            signals.append(StockSignal(
                code=code, name=item["name"],
                signal="buy", reason=f"总分:{item['score']} | {detail}",
                price=item["price"],
                weight=round(weights[i], 4),
            ))

        # ── 卖出信号（跌破MA10 或 止损） ──
        if holdings:
            holdings_set = set(holdings)
            for code in sorted(holdings_set - buy_codes):
                try:
                    stk = sm[code[:2].lower() + code[2:]]
                    k = stk.get_kdata(Query(target_dt - Days(30), target_dt + Days(1)))
                    if k and len(k) >= 10:
                        closes = np.array([float(r.close) for r in k])
                        c = closes[-1]
                        ma10 = np.mean(closes[-10:])
                        buy_price = np.mean(closes[-20:-10]) if len(closes) >= 20 else ma10

                        reasons = []
                        # 跌破 MA10
                        if c < ma10:
                            reasons.append(f"跌破MA10({ma10:.2f})")
                        # 止损
                        loss_pct = (c - buy_price) / buy_price * 100
                        if loss_pct < -stop_loss_pct:
                            reasons.append(f"止损({loss_pct:.1f}%)")

                        if reasons:
                            signals.append(StockSignal(
                                code=code, name=get_stock_display_name(stk),
                                signal="sell", reason="; ".join(reasons),
                                price=c,
                            ))
                        else:
                            signals.append(StockSignal(
                                code=code, name=get_stock_display_name(stk),
                                signal="hold", reason=f"持仓中 MA10={ma10:.2f}",
                                price=c,
                            ))
                except:
                    pass

        # ── 元数据 ──
        avg_scores = {k: round(np.mean([s["factors"][k] for s in selected]), 1) for k in FACTOR_WEIGHTS}
        metadata = {
            "buy_count": len(selected),
            "risk_level": risk_level,
            "risk_summary": self.risk_monitor.summary(target_dt),
            "market_breadth": risk_data["market_breadth"],
            "position_ratio": position_ratio,
            "avg_factors": avg_scores,
            "candidates": len(candidates),
            "scanned": len(stock_codes),
            "stop_loss": stop_loss_pct,
        }

        return StrategyResult(
            date=date_str,
            strategy_name=self.name,
            signals=signals,
            metadata=metadata,
        )
