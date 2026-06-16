"""
市场风险监测模块 — 迁移自聚宽多因子策略

核心功能：
  1. 市场宽度 (market_breadth): 股价在 MA20 以上的个股比例
  2. 行业集中度 (concentration_ratio): 银行/有色/煤炭/钢铁 四大行业 vs 全市场宽度比
  3. 风险等级 (risk_level): 综合判断 0(正常) / 1(警惕) / 2(高风险)
  4. Z-score 偏离度追踪

数据源：Hikyuu（全本地，无需网络）

用法：
    from trade_advisor.strategies.market_risk_monitor import MarketRiskMonitor
    monitor = MarketRiskMonitor()
    risk = monitor.calculate(sm, target_date)
    # → {"risk_level": 0, "market_breadth": 0.72, "concentration_ratio": 0.95, ...}
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from hikyuu.interactive import Datetime, Query, Days


# 四大特殊行业（银行、有色、煤炭、钢铁）
SPECIAL_INDS = {
    'sh801780': '银行I',
    'sh801050': '有色金属I',
    'sh801950': '煤炭I',
    'sh801040': '钢铁I',
}

# 申万一级行业代码（用于行业分类）
SW1_CODES = [
    'sh801010', 'sh801020', 'sh801030', 'sh801040', 'sh801050',
    'sh801060', 'sh801070', 'sh801080', 'sh801110', 'sh801120',
    'sh801150', 'sh801180', 'sh801190', 'sh801200', 'sh801220',
    'sh801230', 'sh801710', 'sh801720', 'sh801730', 'sh801740',
    'sh801750', 'sh801760', 'sh801770', 'sh801780', 'sh801790',
    'sh801880', 'sh801890', 'sh801950', 'sh801960',
]


class MarketRiskMonitor:
    """市场风险监测 — 计算市场宽度、行业集中度、风险等级"""

    def __init__(self, lookback_days: int = 40, history_limit: int = 100):
        self.lookback_days = lookback_days
        self.history_limit = history_limit
        self.risk_history: List[Dict] = []
        self._cache: Dict[str, Dict] = {}
        self._cached_blocks = None
        self._cached_block_date = None

    def _get_industry_stocks(self, sm, industry_code: str) -> list:
        """获取某行业所有成分股（通过申万行业指数获取）"""
        try:
            ind = sm[industry_code]
            blk = sm.get_block_list_by_index_stock(ind)
            return [s.market_code for s in blk]
        except:
            return []

    def _get_stock_industry(self, sm, stock_code: str) -> Optional[str]:
        """
        获取单只股票所属行业
        遍历行业板块，找到包含该股票的行业
        """
        # 缓存行业板块数据
        for ind_code in SW1_CODES:
            try:
                ind = sm[ind_code]
                blk = sm.get_block_list_by_index_stock(ind)
                for s in blk:
                    if s.market_code == stock_code:
                        return ind_code
            except:
                continue
        return None

    def calculate(self, sm, target_date, use_cache: bool = True) -> Dict:
        """
        计算风险指标

        Args:
            sm: Hikyuu 市场对象
            target_date: 目标日期 (hikyuu Datetime 或 str)
            use_cache: 是否使用缓存

        Returns:
            dict: {
                "risk_level": 0-2,
                "concentration_ratio": float,
                "market_breadth": float,
                "z_score": float,
                "special_avg": float,
            }
        """
        # 统一日期格式
        if isinstance(target_date, str):
            parts = target_date.split('-')
            target_dt = Datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        else:
            target_dt = target_date

        date_str = str(target_dt)[:10]

        # 缓存检查
        if use_cache and date_str in self._cache:
            return self._cache[date_str]

        # ── 获取全市场A股（从上证+深证指数成分） ──
        all_stocks = []
        try:
            idx_sh = sm['sh000001']  # 上证指数
            sh_stocks = sm.get_block_list_by_index_stock(idx_sh)
            all_stocks.extend([s.market_code for s in sh_stocks])
        except:
            pass
        try:
            idx_sz = sm['sz399001']  # 深证成指
            sz_stocks = sm.get_block_list_by_index_stock(idx_sz)
            all_stocks.extend([s.market_code for s in sz_stocks])
        except:
            pass

        # 去重、过滤科创/北交所
        filtered_stocks = []
        for code in all_stocks:
            raw = code[2:]  # 去掉 sh/sz 前缀
            if raw[0] in ('4', '8') or raw[:2] == '68':
                continue
            filtered_stocks.append(code)
        all_stocks = list(set(filtered_stocks))

        if len(all_stocks) < 50:
            return self._default_result()

        # ── 获取股价数据 ──
        query_start = target_dt - Days(self.lookback_days + 10)
        query_end = target_dt + Days(1)
        q = Query(query_start, query_end)

        # 采样计算（全市场5000只太慢，取代表性样本）
        sample_size = min(800, len(all_stocks))
        # 按市值排序取前800只（大市值更具代表性）
        sampled_stocks = self._sample_by_cap(sm, all_stocks, sample_size)

        above_ma20_count = 0
        total_valid = 0

        for code in sampled_stocks:
            try:
                stk = sm[code]
                k = stk.get_kdata(q)
                if not k or len(k) < 25:
                    continue
                closes = np.array([float(r.close) for r in k])
                ma20 = np.mean(closes[-20:])
                if closes[-1] > ma20:
                    above_ma20_count += 1
                total_valid += 1
            except:
                continue

        if total_valid < 30:
            return self._default_result()

        market_breadth = above_ma20_count / total_valid

        # ── 行业集中度 ──
        special_vals = []
        for ind_code in SPECIAL_INDS:
            ind_stocks = self._get_industry_stocks(sm, ind_code)
            if not ind_stocks:
                continue
            ind_above = 0
            ind_total = 0
            for code in ind_stocks:
                try:
                    stk = sm[code]
                    k = stk.get_kdata(q)
                    if not k or len(k) < 25:
                        continue
                    closes = np.array([float(r.close) for r in k])
                    ma20 = np.mean(closes[-20:])
                    if closes[-1] > ma20:
                        ind_above += 1
                    ind_total += 1
                except:
                    continue
            if ind_total > 0:
                special_vals.append(ind_above / ind_total)

        special_avg = np.mean(special_vals) if special_vals else 0.5
        concentration_ratio = special_avg / (market_breadth + 1e-5)

        # ── Z-score（过去20天偏离度） ──
        if len(self.risk_history) >= 20:
            hist_special = [r['special_avg'] for r in self.risk_history[-20:]]
            mean_s = np.mean(hist_special)
            std_s = np.std(hist_special) + 1e-5
            z_score = (special_avg - mean_s) / std_s
        else:
            z_score = 0.0

        # ── 风险等级 ──
        if z_score > 1.0 and concentration_ratio > 1.2 and market_breadth < 0.6:
            risk_level = 2
        elif (z_score > 0.8 and concentration_ratio > 1.1) or market_breadth < 0.5:
            risk_level = 1
        else:
            risk_level = 0

        # ── 结果 ──
        result = {
            'date': date_str,
            'risk_level': risk_level,
            'concentration_ratio': round(concentration_ratio, 3),
            'market_breadth': round(market_breadth, 3),
            'special_avg': round(special_avg, 3),
            'z_score': round(z_score, 3),
            'sample_count': total_valid,
        }

        # 记录历史
        self.risk_history.append(result)
        self._cache[date_str] = result

        # 限制历史长度
        if len(self.risk_history) > self.history_limit:
            self.risk_history = self.risk_history[-self.history_limit:]
        if len(self._cache) > 30:
            keys = sorted(self._cache.keys())[:-30]
            for k in keys:
                self._cache.pop(k, None)

        return result

    def _sample_by_cap(self, sm, stocks: List[str], n: int) -> List[str]:
        """按流通市值排序采样前 N 只"""
        cap_sorted = []
        for code in stocks:
            try:
                stk = sm[code]
                fin = stk.get_finance_info()
                if fin and fin.have('liutongguben'):
                    cap_sorted.append((code, float(fin['liutongguben'])))
            except:
                continue
        cap_sorted.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in cap_sorted[:n]]

    def _default_result(self) -> Dict:
        """数据不足时的默认返回值"""
        return {
            'risk_level': 0,
            'concentration_ratio': 0,
            'market_breadth': 0.5,
            'special_avg': 0.5,
            'z_score': 0,
            'sample_count': 0,
        }

    def get_risk_label(self, risk_level: int) -> str:
        labels = {0: '🟢 正常', 1: '🟡 警惕', 2: '🔴 高风险'}
        return labels.get(risk_level, '未知')

    def summary(self, target_date=None) -> str:
        """生成风险摘要报告"""
        latest = self.risk_history[-1] if self.risk_history else None
        if latest is None:
            return "暂无风险数据"

        return (
            f"📊 市场风险监测 [{latest['date']}]\n"
            f"  • 风险等级: {self.get_risk_label(latest['risk_level'])} (级别{latest['risk_level']})\n"
            f"  • 市场宽度: {latest['market_breadth']:.1%} "
            f"(>MA20比例)\n"
            f"  • 行业集中度: {latest['concentration_ratio']:.2f} "
            f"(四大行业/全市场)\n"
            f"  • Z-score: {latest['z_score']:+.2f} "
            f"(偏离度)\n"
            f"  • 样本数量: {latest['sample_count']} 只"
        )


if __name__ == '__main__':
    from hikyuu.interactive import *
    monitor = MarketRiskMonitor()
    result = monitor.calculate(sm, Datetime.now())
    print(monitor.summary())
