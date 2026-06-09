"""
A股策略决策系统 — 启动入口

用法:
    conda activate stock
    python -m trade_advisor.main
"""

import sys
import os

# 确保能找到 hikyuu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_advisor import ui

if __name__ == "__main__":
    ui.run()
