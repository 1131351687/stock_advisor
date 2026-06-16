"""
数据适配层 - 混合数据源 (Hikyuu 主 + AKShare 备选)

Hikyuu 优势：快速本地库，适合 A 股数据
AKShare 优势：云 API，支持特殊数据（板块、行情补充）

使用方式：
    df = get_kdata('sz000001', '2024-01-01', '2024-06-09')  # 自动选择最佳源
    stocks = get_stock_list('小盘')  # 获取股票列表
    cap = get_market_cap('sz000001', '2024-06-01')  # 获取流通市值
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Hikyuu 数据源
from hikyuu.interactive import *

# 可选：AKShare 备选数据源
try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False
    ak = None

# 配置日志
logger = logging.getLogger(__name__)


# ============================================================================
# Hikyuu 数据获取函数
# ============================================================================

def get_kdata_from_hikyuu(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 Hikyuu 获取 K 线数据

    Args:
        code: 股票代码，格式 'sh000001' 或 'sz000001'
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'

    Returns:
        DataFrame，列：['date', 'open', 'high', 'low', 'close', 'volume']
        按日期升序排列（便于指标计算）
    """
    try:
        # 标准化代码格式
        if len(code) == 6:  # 如果只是数字代码，自动补市场前缀
            code = ('sh' if code.startswith('6') else 'sz') + code

        stock = sm[code.lower()]

        # 构建查询对象
        start_parts = start_date.split('-')
        end_parts = end_date.split('-')
        start_dt = Datetime(int(start_parts[0]), int(start_parts[1]), int(start_parts[2]))
        end_dt = Datetime(int(end_parts[0]), int(end_parts[1]), int(end_parts[2]))

        query = Query(start_dt, end_dt + Days(1))
        kdata = stock.get_kdata(query)

        if not kdata or len(kdata) == 0:
            logger.warning(f"Hikyuu: {code} 无 K 线数据 ({start_date} 至 {end_date})")
            return pd.DataFrame()

        # 转换为 DataFrame
        records = []
        for k in kdata:
            dt = k.datetime.datetime()
            records.append({
                'date': dt,
                'open': float(k.open),
                'high': float(k.high),
                'low': float(k.low),
                'close': float(k.close),
                'volume': float(k.volume),
            })

        df = pd.DataFrame(records)
        df = df.sort_values('date', ascending=True).reset_index(drop=True)

        logger.info(f"Hikyuu: 获取 {code} {len(df)} 条 K 线数据")
        return df

    except Exception as e:
        logger.warning(f"Hikyuu 读取失败 ({code}): {e}")
        return pd.DataFrame()


def get_stock_list_from_hikyuu(category: str = 'all') -> List[str]:
    """
    从 Hikyuu 获取股票列表

    get_block_list_by_index_stock() 返回 Block 对象列表，
    每个 Block 通过 get_stock_list() 获取 Stock 对象列表。

    Args:
        category: 分类
            - 'all': 全市场
            - 'a': A 股
            - 'small_cap': 中小板指成分 (399101)
            - 'tech': 科技板块（从创业板获取代理）

    Returns:
        股票代码列表 ['sh600000', 'sz000001', ...]
    """
    try:
        raw_stocks = []  # Stock 对象列表

        if category == 'all' or category == 'a':
            # 获取全市场 A 股（通过主要指数成分）
            for idx_code in ['sh000001', 'sz000001']:  # 上证指数、深证成指
                idx = sm[idx_code]
                blocks = sm.get_block_list_by_index_stock(idx)
                for blk in blocks:
                    for s in blk.get_stock_list():
                        raw_stocks.append(s)

        elif category == 'small_cap':
            # 中小板指 (399101)
            idx = sm['sz399101']
            blocks = sm.get_block_list_by_index_stock(idx)
            if blocks and len(blocks) > 0:
                raw_stocks = blocks[0].get_stock_list()

        elif category == 'tech':
            # 创业板指 (399006) 作为科技板块代理
            idx = sm['sz399006']
            blocks = sm.get_block_list_by_index_stock(idx)
            if blocks and len(blocks) > 0:
                raw_stocks = blocks[0].get_stock_list()

        else:
            logger.warning(f"未知分类: {category}")

        # 转为 market_code 格式 (如 'sh600000')
        codes = []
        for s in raw_stocks:
            try:
                mc = s.market_code.lower()
                codes.append(mc)
            except:
                continue

        # 去重
        codes = list(set(codes))
        logger.info(f"Hikyuu: 获取 {category} 股票池 {len(codes)} 只")
        return sorted(codes)

    except Exception as e:
        logger.warning(f"Hikyuu 获取股票列表失败: {e}")
        return []


def get_market_cap_from_hikyuu(code: str, date: str = None) -> float:
    """
    从 Hikyuu SQLite 数据库获取流通市值 (亿元)

    Args:
        code: 股票代码
        date: 查询日期（不使用）

    Returns:
        流通市值 (亿元)，失败返回 0
    """
    try:
        if len(code) == 6:
            code = ('sh' if code.startswith('6') else 'sz') + code

        stock = sm[code.lower()]
        finance = stock.get_finance_info()

        # 流通股本(股) → 流通市值(亿元) = 流通股本 * 收盘价 / 1亿
        if finance and finance.have('liutongguben'):
            liutongguben = float(finance['liutongguben'])  # 股
            # 获取最新收盘价
            k = stock.get_kdata(Query(-1))
            if k and len(k) > 0:
                close = float(k[-1].close)
                market_cap_yi = liutongguben * close / 1e8  # 亿元
                return market_cap_yi

        return 0.0

    except Exception as e:
        logger.warning(f"获取流通市值失败 ({code}): {e}")
        return 0.0


# ============================================================================
# AKShare 数据获取函数（备选）
# ============================================================================

def get_kdata_from_akshare(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 AKShare 获取 K 线数据（备选方案）

    Args:
        code: 股票代码 'sz000001' 或 '000001'
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'

    Returns:
        DataFrame，列：['date', 'open', 'high', 'low', 'close', 'volume']
    """
    if not HAS_AKSHARE:
        logger.warning("AKShare 未安装，无法使用备选数据源")
        return pd.DataFrame()

    try:
        # 标准化代码格式 (AKShare 需要格式 'sz000001' 或 'sh600000')
        if len(code) == 6:
            code = ('sh' if code.startswith('6') else 'sz') + code
        elif code.startswith('sz') or code.startswith('sh'):
            pass  # 已有市场前缀
        else:
            code = ('sh' if code.startswith('6') else 'sz') + code

        # 调用 AKShare
        df = ak.stock_zh_a_hist(
            symbol=code[2:],  # 去掉市场前缀
            period='daily',
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', '')
        )

        if df.empty:
            logger.warning(f"AKShare: {code} 无数据")
            return pd.DataFrame()

        # 标准化列名
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
        })

        # 转换数据类型
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.sort_values('date', ascending=True).reset_index(drop=True)

        logger.info(f"AKShare: 获取 {code} {len(df)} 条 K 线数据")
        return df

    except Exception as e:
        logger.warning(f"AKShare 读取失败 ({code}): {e}")
        return pd.DataFrame()


# ============================================================================
# 统一接口 - 混合数据源
# ============================================================================

def get_kdata(
    code: str,
    start_date: str,
    end_date: str,
    source: str = 'auto',
    fallback: bool = True
) -> pd.DataFrame:
    """
    获取 K 线数据 - 智能选择数据源

    Args:
        code: 股票代码 'sz000001' 或 '000001'
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        source: 数据源选择
            - 'auto': 优先 Hikyuu，失败则 AKShare（推荐）
            - 'hikyuu': 仅 Hikyuu
            - 'akshare': 仅 AKShare
        fallback: 是否允许自动降级到备选源

    Returns:
        DataFrame，列：['date', 'open', 'high', 'low', 'close', 'volume']
    """

    # 优先尝试 Hikyuu（本地，快速）
    if source in ['auto', 'hikyuu']:
        df = get_kdata_from_hikyuu(code, start_date, end_date)
        if not df.empty:
            return df

        if source == 'hikyuu':
            return df  # 仅 Hikyuu，返回空

    # 降级到 AKShare（网络，较慢但完整）
    if fallback and source != 'hikyuu':
        logger.info(f"Hikyuu 数据不可用，尝试 AKShare ({code})")
        return get_kdata_from_akshare(code, start_date, end_date)

    return pd.DataFrame()


def get_stock_list(category: str = 'all') -> List[str]:
    """
    获取股票列表（优先 Hikyuu）

    Args:
        category: 分类 ('all', 'small_cap', 'tech', 等)

    Returns:
        股票代码列表
    """
    try:
        stocks = get_stock_list_from_hikyuu(category)
        if stocks:
            return stocks
    except Exception as e:
        logger.warning(f"Hikyuu 获取股票列表失败: {e}")

    # AKShare 备选
    if HAS_AKSHARE and category == 'all':
        try:
            logger.info("使用 AKShare 获取全市场股票列表...")
            df = ak.stock_zh_a_spot()
            if not df.empty:
                return df['代码'].tolist()
        except Exception as e:
            logger.warning(f"AKShare 备选失败: {e}")

    return []


def get_market_cap(code: str, date: str = None) -> float:
    """
    获取流通市值 (亿元)

    Args:
        code: 股票代码
        date: 查询日期（可选）

    Returns:
        流通市值 (亿元)
    """
    return get_market_cap_from_hikyuu(code, date)


def get_block_list(block_code: str = 'sz399101') -> List[str]:
    """
    获取板块或指数成分股

    Args:
        block_code: 板块/指数代码，如 'sz399101' (中小板指)

    Returns:
        股票代码列表
    """
    try:
        if len(block_code) == 6:  # 标准化格式
            block_code = ('sh' if block_code.startswith('6') else 'sz') + block_code

        idx = sm[block_code.lower()]
        blocks = sm.get_block_list_by_index_stock(idx)
        codes = []
        for blk in blocks:
            for s in blk.get_stock_list():
                try:
                    codes.append(s.market_code.lower())
                except:
                    continue

        logger.info(f"获取 {block_code} 成分股 {len(codes)} 只")
        return list(set(codes))

    except Exception as e:
        logger.warning(f"获取板块成分失败 ({block_code}): {e}")
        return []


# ============================================================================
# 数据检查函数
# ============================================================================

def validate_kdata(df: pd.DataFrame, min_rows: int = 20) -> Tuple[bool, str]:
    """
    验证 K 线数据完整性

    Args:
        df: K 线 DataFrame
        min_rows: 最小行数要求

    Returns:
        (是否有效, 错误信息)
    """
    if df.empty:
        return False, "数据为空"

    if len(df) < min_rows:
        return False, f"数据不足 {min_rows} 行"

    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"缺少列: {missing}"

    if df['close'].isna().any():
        return False, "存在 NaN 值"

    return True, ""


# ============================================================================
# 测试函数
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\n=== 测试数据适配层 ===\n")

    # 测试 1: 获取 K 线数据
    print("测试 1: 获取 K 线数据 (Hikyuu)")
    df = get_kdata('sz000001', '2024-01-01', '2024-06-09')
    print(f"  获取 {len(df)} 条数据")
    if not df.empty:
        print(f"  日期范围: {df['date'].min()} 至 {df['date'].max()}")
        print(df.head(3))

    # 测试 2: 获取股票列表
    print("\n测试 2: 获取股票列表")
    stocks = get_stock_list('small_cap')
    print(f"  小盘股: {len(stocks)} 只")
    if stocks:
        print(f"  示例: {stocks[:5]}")

    # 测试 3: 获取流通市值
    print("\n测试 3: 获取流通市值")
    cap = get_market_cap('sh600519')  # 贵州茅台
    print(f"  茅台流通市值: {cap:.2f} 亿元")

    # 测试 4: 获取板块成分
    print("\n测试 4: 获取板块成分")
    codes = get_block_list('sz399101')  # 中小板指
    print(f"  中小板指成分: {len(codes)} 只")
    if codes:
        print(f"  示例: {codes[:5]}")

    print("\n✓ 数据适配层测试完成")
