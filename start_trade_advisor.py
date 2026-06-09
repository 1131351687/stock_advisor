"""
A股策略决策系统 — 启动入口（带日志）
"""
import sys
import os
import traceback

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "启动日志.txt")

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        from datetime import datetime
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

# 清空旧日志后写入新日志头
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")
log("=== 启动 ===")

try:
    # 添加项目路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONIOENCODING"] = "utf-8"

    log("初始化 Hikyuu...")
    from hikyuu.interactive import *
    log("Hikyuu 初始化完成")

    log("加载图形界面...")
    from trade_advisor.ui import run
    log("启动界面...")
    run()
    log("程序正常退出")

except Exception as e:
    err = f"!! {e}\n{traceback.format_exc()}"
    log(err)
    # 也弹窗显示错误
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, err, "A股策略决策系统 - 启动失败", 0x10)
    except:
        pass
    sys.exit(1)
