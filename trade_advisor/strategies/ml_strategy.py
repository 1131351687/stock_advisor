"""
ML 因子策略 — 使用预训练的 LightGBM 模型进行预测评分

工作流程：
  1. 加载已训练的模型 (models/ml_model.pkl)
  2. 对候选股票计算技术因子
  3. 用模型预测综合评分
  4. 评分排序选股
  5. 跌破 MA10 卖出

训练模型：
  conda activate stock
  python -m trade_advisor.ml_trainer     # 首次训练
  python -m trade_advisor.ml_trainer --force  # 重新训练
"""

import numpy as np
import logging
from typing import Optional, Dict, List

from trade_advisor.strategies import (
    BaseStrategy, StrategyParam, StockSignal, StrategyResult,
    _to_hikyuu_date, _to_date_str, _get_last_trade_date,
    get_stock_display_name,
)
from hikyuu.interactive import *
from trade_advisor.data_adapter import get_kdata, get_stock_list

from trade_advisor.ml_trainer import load_models, predict_score, compute_features

logger = logging.getLogger(__name__)


class MLFactorStrategy(BaseStrategy):
    """ML 因子策略 — 使用预训练的 LightGBM 模型评分"""

    def __init__(self):
        super().__init__()
        self._model = None

    @property
    def name(self) -> str:
        return "ML 因子策略"

    @property
    def description(self) -> str:
        return "LightGBM 三模型评分（分类+回归+方向），需先运行 ml_trainer 训练"

    @property
    def params(self) -> list:
        return [
            StrategyParam("buy_count", "持仓数量", 5, "int", min_val=1, max_val=20),
            StrategyParam("score_threshold", "评分阈值", 30, "int", min_val=10, max_val=90),
            StrategyParam("max_stocks", "扫描数量", 300, "int", min_val=50, max_val=1000),
            StrategyParam("stop_loss", "止损(%)", 8, "float", min_val=3, max_val=20, step=1),
            StrategyParam("lookback_days", "回溯天数", 100, "int", min_val=60, max_val=200),
        ]

    def _ensure_model(self) -> bool:
        """确保模型已加载"""
        if self._model is not None:
            return True
        self._model = load_models()
        return self._model is not None

    def run(self, param_values: dict, date=None, holdings=None) -> StrategyResult:
        # 检查模型
        if not self._ensure_model():
            return StrategyResult(
                date=_to_date_str(_to_hikyuu_date(date) if date else _get_last_trade_date()[1] if _get_last_trade_date()[1] != "N/A" else "N/A"),
                strategy_name=self.name,
                signals=[],
                metadata={"error": "模型未训练，请先执行: python -m trade_advisor.ml_trainer"},
            )

        buy_count = int(param_values.get("buy_count", 5))
        score_threshold = int(param_values.get("score_threshold", 30))
        max_stocks = int(param_values.get("max_stocks", 300))
        lookback_days = int(param_values.get("lookback_days", 100))

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

        # 数据存在检查
        _ref = sm["sh510050"].get_kdata(Query(target_dt, target_dt + Days(1)))
        if not _ref or len(_ref) == 0:
            _, fb = _get_last_trade_date()
            if fb and fb != "N/A":
                target_dt = _to_hikyuu_date(fb)
                date_str = fb

        # ── 获取股票池 ──
        stock_codes = get_stock_list('small_cap')
        if len(stock_codes) < 50:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                 metadata={"error": "股票池不足"})

        # ── ML 预测评分 ──
        query_start = (target_dt - Days(lookback_days + 20)).datetime().date()
        query_end = target_dt.datetime().date()

        candidates = []
        for i, code in enumerate(stock_codes[:max_stocks]):
            try:
                df = get_kdata(code, str(query_start), str(query_end))
                if df.empty or len(df) < 60:
                    continue

                # ML 评分
                score = predict_score(self._model, df)
                if score < score_threshold:
                    continue

                stk = sm[code.lower()]
                name = get_stock_display_name(stk)
                close = float(df['close'].values[-1])

                candidates.append({
                    "code": code,
                    "name": name,
                    "score": score,
                    "price": close,
                })
            except:
                continue

        # ── 排序选股 ──
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:buy_count]

        if not selected:
            return StrategyResult(date=date_str, strategy_name=self.name,
                                  signals=[],
                                  metadata={"error": f"ML评分均低于{score_threshold}",
                                            "scanned": min(max_stocks, len(stock_codes))})

        # ── 信号 ──
        signals = []
        buy_codes = set()
        for item in selected:
            code = item["code"]
            buy_codes.add(code)
            signals.append(StockSignal(
                code=code, name=item["name"],
                signal="buy",
                reason=f"ML评分:{item['score']:.1f}",
                price=item["price"],
                weight=round(1.0 / len(selected), 4),
            ))

        # ── 卖出：跌破 MA10 ──
        if holdings:
            for code in sorted(set(holdings) - buy_codes):
                try:
                    stk = sm[code[:2].lower() + code[2:]]
                    k = stk.get_kdata(Query(target_dt - Days(30), target_dt + Days(1)))
                    if k and len(k) >= 10:
                        closes = np.array([float(r.close) for r in k])
                        ma10 = np.mean(closes[-10:])
                        if closes[-1] < ma10:
                            signals.append(StockSignal(
                                code=code, name=get_stock_display_name(stk),
                                signal="sell", reason=f"跌破MA10({ma10:.2f})",
                                price=float(closes[-1]),
                            ))
                except:
                    pass

        return StrategyResult(
            date=date_str,
            strategy_name=self.name,
            signals=signals,
            metadata={
                "buy_count": len(selected),
                "candidates": len(candidates),
                "scanned": min(max_stocks, len(stock_codes)),
                "avg_score": round(np.mean([s["score"] for s in selected]), 1),
            },
        )
