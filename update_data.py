"""
Hikyuu 数据更新工具
一键更新: K线日线、股票名称、财务数据、板块成分

用法:
  conda activate stock
  python update_data.py
"""
import sys
import os
import time
import sqlite3

DEST_DIR = "D:/tools/Hikyuu"
STOCK_DB = os.path.join(DEST_DIR, "stock.db")

# 多组通达信服务器地址（依次尝试）
TDX_SERVERS = [
    ("180.101.48.170", 7709),
    ("180.153.39.51", 7709),
    ("119.147.212.101", 7709),
    ("180.101.48.171", 7709),
    ("119.147.212.81", 7709),
]


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}")
    sys.stdout.flush()


def connect_tdx():
    """尝试连接通达信，返回 api 实例"""
    from pytdx.hq import TdxHq_API

    # 1. 尝试使用 pytdx 自带的寻优选择最快服务器
    try:
        log("  正在获取最优行情服务器 (select_best_ip) ...")
        from pytdx.util.best_ip import select_best_ip
        best_ip = select_best_ip()
        if best_ip and 'ip' in best_ip and 'port' in best_ip:
            ip, port = best_ip['ip'], best_ip['port']
            log(f"  + 找到最优服务器: {ip}:{port}")
            api = TdxHq_API()
            if api.connect(ip, port):
                log(f"  + 连接成功 ({ip}:{port})")
                return api
    except Exception as e:
        log(f"  自动获取最优服务器失败: {e}，将尝试备用服务器列表")

    # 2. 备用静态列表（包含目前已知的可用好服务器）
    fallback_servers = [
        ("60.12.136.250", 7709),
        ("115.238.56.198", 7709),
        ("115.238.90.165", 7709),
        ("180.153.18.170", 7709),
        ("218.75.126.9", 7709),
        ("60.191.117.167", 7709),
        ("jstdx.gtjas.com", 7709),
        ("shtdx.gtjas.com", 7709),
        ("sztdx.gtjas.com", 7709),
    ] + TDX_SERVERS

    for ip, port in fallback_servers:
        log(f"  尝试连接备用 {ip}:{port} ...")
        api = TdxHq_API()
        try:
            if api.connect(ip, port):
                log(f"  + 连接成功 ({ip}:{port})")
                return api
            api.disconnect()
        except Exception:
            try:
                api.disconnect()
            except Exception:
                pass
    return None


def verify_data():
    """验证更新后的数据是否可用"""
    log("\n" + "-" * 40)
    log("验证数据...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        # 轻量检查：直接读 HDF5 文件的最新日期
        import h5py
        import numpy as np

        for fname, label in [("sh_day.h5", "上证"), ("sz_day.h5", "深证")]:
            fpath = os.path.join(DEST_DIR, fname)
            if not os.path.exists(fpath):
                log(f"  {label} 数据文件不存在: {fpath}")
                continue
            f = h5py.File(fpath, "r")
            grp = f.get("data")
            if grp is None:
                log(f"  {label} HDF5 缺少 data 组")
                f.close()
                continue
            # 取第一个有数据的标的，看最新日期
            latest_dates = []
            for stk_code in list(grp.keys())[:50]:  # 只看前50只
                ds = grp[stk_code]
                if len(ds) > 0:
                    dt = ds[-1]["datetime"]
                    y = dt // 10000000000
                    m = (dt // 100000000) % 100
                    d = (dt // 1000000) % 100
                    latest_dates.append(f"{int(y):04d}-{int(m):02d}-{int(d):02d}")
            f.close()
            if latest_dates:
                # 去重取最晚的
                unique = sorted(set(latest_dates), reverse=True)
                log(f"  {label} 最新数据日期范围: {unique[0]} ~ {unique[-1]}")
                log(f"  {label} 抽样 {len(latest_dates)} 只ETF中，")
                log(f"    最新日期: {unique[0]}")
                log(f"    最旧日期: {unique[-1]}")
        return True
    except Exception as e:
        log(f"  [跳过] 数据验证失败: {e}")
        return False


def main():
    log("=" * 50)
    log("Hikyuu 数据更新工具")
    log(f"数据目录: {DEST_DIR}")
    log("=" * 50)

    if not os.path.exists(STOCK_DB):
        log(f"[错误] 找不到 stock.db: {STOCK_DB}")
        log("请先确认 Hikyuu 已初始化")
        sys.exit(1)

    from hikyuu.data.pytdx_to_h5 import (
        import_data, import_stock_name, import_index_name
    )
    from hikyuu.data.pytdx_finance_to_sqlite import pytdx_import_finance_to_sqlite

    # 连接数据库
    connect = sqlite3.connect(STOCK_DB)

    # 连接通达信
    log("\n连接通达信行情服务器...")
    api = connect_tdx()
    if api is None:
        log("[错误] 所有服务器均无法连接，请检查网络")
        connect.close()
        sys.exit(1)

    quotations = ["stock"]

    # ── 1. 更新指数代码表 ──
    log("\n[1/5] 更新指数代码表...")
    try:
        n = import_index_name(connect)
        log(f"  + 指数个数: {n}")
    except Exception as e:
        log(f"  [跳过] {e}")

    # ── 2. 更新股票名称与状态 ──
    log("\n[2/5] 更新股票名称与上市状态...")
    for mkt in ("SH", "SZ"):
        try:
            n = import_stock_name(connect, api, mkt, quotations)
            log(f"  + {mkt}: 处理 {n} 只")
        except Exception as e:
            log(f"  [跳过] {mkt}: {e}")

    # ── 3. 更新K线日线数据 ──
    log("\n[3/5] 更新 K 线日线数据（核心）...")
    for mkt in ("SH", "SZ", "BJ"):
        try:
            log(f"  正在导入 {mkt} 日线...")
            t0 = time.time()
            n = import_data(connect, mkt, "DAY", quotations, api, DEST_DIR)
            elapsed = time.time() - t0
            log(f"  + {mkt} 新增 {n} 条记录 (耗时 {elapsed:.0f}s)")
        except Exception as e:
            log(f"  [跳过] {mkt}: {e}")

    # ── 4. 更新财务数据 ──
    log("\n[4/5] 更新财务数据（流通股本等）...")
    for mkt in ("SH", "SZ"):
        try:
            n = pytdx_import_finance_to_sqlite(connect, api, mkt)
            log(f"  + {mkt} 新增 {n} 条")
        except Exception as e:
            log(f"  [跳过] {mkt}: {e}")

    # ── 5. 更新板块数据 ──
    log("\n[5/5] 更新板块成分数据...")
    try:
        from hikyuu.data.download_block import download_block_info
        download_block_info()
        log(f"  + 板块数据更新完成")
    except Exception as e:
        log(f"  [跳过] {e}")

    api.disconnect()
    connect.close()

    # ── 验证 ──
    verify_data()

    # 日期提示
    log("\n" + "=" * 50)
    log("更新完成！")
    log("重新打开策略决策系统即可使用最新数据")
    log("=" * 50)


if __name__ == "__main__":
    main()
