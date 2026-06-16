"""
数据库状态检测工具

检测 Hikyuu HDF5 数据文件的最新日期，判断数据库是否已更新到最新。

用法:
    from trade_advisor.data_status import get_db_latest_dates, get_data_freshness

    dates = get_db_latest_dates()
    # → {'sh': '2024-06-07', 'sz': '2024-06-07', 'overall': '2024-06-07'}

    status = get_data_freshness()
    # → {'is_fresh': True, 'latest_date': '2024-06-07', 'days_diff': 2, 'status_label': '🟢 正常'}
"""

import os
import sys
import h5py
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple

# Hikyuu 数据目录（与 update_data.py 同步）
DEST_DIR = "D:/tools/Hikyuu"

# 标记一周内无数据为"正常"（交易日间隔）
FRESHNESS_THRESHOLD_DAYS = 7


def _parse_hdf5_datetime(dt_int: int) -> str:
    """
    解析 HDF5 中的 datetime 整数为日期字符串

    HDF5 存储格式: YYYYMMDDHHMMSS (如 20240607153000)
    """
    s = str(int(dt_int))
    if len(s) >= 8:
        y, m, d = s[:4], s[4:6], s[6:8]
        return f"{y}-{m}-{d}"
    return "未知"


def get_latest_date_from_hdf5(filepath: str, max_check: int = 100) -> Optional[str]:
    """
    从 HDF5 文件中读取最新数据日期

    Args:
        filepath: HDF5 文件路径 (如 sh_day.h5 或 sz_day.h5)
        max_check: 最多检查的前 N 只股票

    Returns:
        最新日期字符串 'YYYY-MM-DD'，或在失败时返回 None
    """
    if not os.path.exists(filepath):
        return None

    try:
        f = h5py.File(filepath, "r")
        grp = f.get("data")
        if grp is None:
            f.close()
            return None

        latest_found = None
        codes = list(grp.keys())

        for stk_code in codes[:max_check]:
            ds = grp[stk_code]
            if ds is None or len(ds) == 0:
                continue

            # 取该只股票的最新日期（最后一条记录）
            dt = ds[-1]["datetime"]
            date_str = _parse_hdf5_datetime(dt)

            if latest_found is None or date_str > latest_found:
                latest_found = date_str

        f.close()
        return latest_found

    except Exception as e:
        print(f"[data_status] 读取 HDF5 失败 ({filepath}): {e}")
        return None


def get_db_latest_dates() -> Dict[str, str]:
    """
    获取数据库最新日期信息

    Returns:
        dict:
            sh: 上证最新日期
            sz: 深证最新日期
            overall: 全局最新日期（两市中较新的）
            sh_file: 上证文件是否存在
            sz_file: 深证文件是否存在
    """
    sh_path = os.path.join(DEST_DIR, "sh_day.h5")
    sz_path = os.path.join(DEST_DIR, "sz_day.h5")

    sh_latest = get_latest_date_from_hdf5(sh_path)
    sz_latest = get_latest_date_from_hdf5(sz_path)

    # 取最晚的日期
    overall = None
    for d in [sh_latest, sz_latest]:
        if d is not None:
            if overall is None or d > overall:
                overall = d

    return {
        "sh": sh_latest or "无数据",
        "sz": sz_latest or "无数据",
        "overall": overall or "无数据",
        "sh_file": str(os.path.exists(sh_path)),
        "sz_file": str(os.path.exists(sz_path)),
    }


def get_data_freshness() -> Dict:
    """
    判断数据的时效性

    Returns:
        dict:
            latest_date: 数据库中最新的日期
            today: 今天日期
            days_diff: 距离今天的天数
            is_fresh: 是否在阈值之内
            status_label: 中文状态标签（含图标）
            status_color: 状态颜色
    """
    dates = get_db_latest_dates()
    latest = dates.get("overall", "无数据")

    today = datetime.now().strftime("%Y-%m-%d")

    if latest == "无数据":
        return {
            "latest_date": "无数据",
            "today": today,
            "days_diff": -1,
            "is_fresh": False,
            "status_label": "🔴 无数据，请更新",
            "status_color": "#e74c3c",
        }

    # 计算天数差
    try:
        latest_dt = datetime.strptime(latest, "%Y-%m-%d")
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        days_diff = (today_dt - latest_dt).days
    except:
        days_diff = 999

    if days_diff <= 0:
        # 当天数据
        status = "✅ 今日已更新"
        color = "#27ae60"
        is_fresh = True
    elif days_diff <= 2:
        # 1-2 天前（含周末，正常）
        status = "🟢 数据较新"
        color = "#27ae60"
        is_fresh = True
    elif days_diff <= FRESHNESS_THRESHOLD_DAYS:
        # 3-7 天
        status = f"🟡 {days_diff}天前更新"
        color = "#f39c12"
        is_fresh = True
    else:
        # 超过阈值
        status = f"🔴 {days_diff}天未更新"
        color = "#e74c3c"
        is_fresh = False

    return {
        "latest_date": latest,
        "today": today,
        "days_diff": days_diff,
        "is_fresh": is_fresh,
        "status_label": status,
        "status_color": color,
    }


def run_update(log_func=print) -> bool:
    """
    运行数据更新（阻塞调用）

    Args:
        log_func: 日志输出函数（默认 print）

    Returns:
        是否成功
    """
    try:
        # 调用 update_data.py 的 main 函数
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 重定向 stdout 到 log_func
        import update_data
        original_stdout = sys.stdout

        class LogRedirect:
            def write(self, msg):
                if msg.strip():
                    log_func(msg.strip())
            def flush(self):
                pass

        sys.stdout = LogRedirect()
        update_data.main()
        sys.stdout = original_stdout
        return True
    except Exception as e:
        log_func(f"更新失败: {e}")
        return False
    finally:
        sys.stdout = original_stdout


if __name__ == "__main__":
    # 测试
    print("=== 数据库状态检测 ===\n")
    dates = get_db_latest_dates()
    print(f"上证数据: {dates['sh']}")
    print(f"深证数据: {dates['sz']}")
    print(f"全局最新: {dates['overall']}")

    print("\n--- 时效性 ---")
    freshness = get_data_freshness()
    print(f"状态: {freshness['status_label']}")
    print(f"最新日期: {freshness['latest_date']}")
    print(f"差异天数: {freshness['days_diff']}")
    print(f"数据较新: {freshness['is_fresh']}")
