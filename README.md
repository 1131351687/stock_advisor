# Stock Advisor — A 股量化策略平台

基于 Hikyuu 本地数据库的量化交易研究与辅助决策系统，集策略开发、回测验证、实时选股、仓位管理于一体，支持多类选股策略与图形化操作界面。

---

## 特性

**混合数据源** — 主数据源使用 Hikyuu 本地 HDF5 + SQLite 数据库，低延迟、高可靠；当本地数据缺失时自动降级至 AKShare 网络 API，确保策略不因数据中断而停摆。

**策略库丰富** — 系统内置多类选股策略，覆盖小市值、技术形态、动量轮动、多因子评分、机器学习等方向，策略间可横向对比。

**完整回测引擎** — 支持日/周/月调仓周期，内置手续费、印花税成本模型，输出年化收益率、最大回撤、夏普比率、月度收益分布等绩效指标。小市值和 ETF 策略拥有专用加速回测路径。

**图形化界面** — 基于 PySide6 构建的桌面 GUI，支持策略参数实时调节、一键运行选股、决策记录浏览与回溯。

**数据持久化** — 所有历史决策存入 SQLite 数据库（`decisions.db`），支持按日期、策略查询和导出，便于事后归因分析。

**风险自适应** — 增强多因子策略内置三级风控体系：市场宽度监测、行业集中度分析、Z-score 异常检测，中风险降仓位、高风险切换至防御 ETF。

---

## 快速开始

### 环境要求

- Python 3.8+
- Hikyuu 库及本地数据库（HDF5 + SQLite）
- 依赖详见 `requirements.txt`

### 安装

```powershell
pip install -r requirements.txt
python -c "import hikyuu; print(hikyuu.__file__)"  # 验证安装
```

### 数据准备

1. 按 Hikyuu 官方说明初始化本地数据库（默认目录：`D:/tools/Hikyuu`）
2. 确保 `stock.db` 和 `sh_day.h5` / `sz_day.h5` 已就绪
3. 运行数据更新：`python update_data.py`

### 启动

```powershell
python start_trade_advisor.py    # 图形界面
python -m trade_advisor.main     # 或直接运行模块
```

---

## 与 Hikyuu 数据源的集成

本项目深度整合 Hikyuu 作为核心数据引擎，所有策略均通过 Hikyuu 的 C++ 高性能后端获取行情与财务数据。集成方式如下：

### 全局导入

项目在入口处通过 `from hikyuu.interactive import *` 一次性导入 Hikyuu 上下文，之后所有模块可直接使用 `sm`（StockManager 单例）访问数据：

```python
from hikyuu.interactive import *

# sm 是全局 StockManager 实例
stk = sm["sh510050"]                        # 按代码获取 Stock 对象
k = stk.get_kdata(Query(-100))               # 获取最近 100 条 K 线
blocks = sm.get_block_list_by_index_stock(idx)  # 获取指数成分股
cal = sm.get_trading_calendar(query)         # 获取交易日历
```

### 核心 Hikyuu API 调用方式

```python
# ── K 线数据 ──
stock = sm["sz000001"]                                          # 平安银行
query = Query(Datetime(2024, 1, 1), Datetime(2024, 6, 9) + Days(1))
kdata = stock.get_kdata(query)                                   # 返回 KData 对象
for k in kdata:
    print(k.datetime, k.open, k.high, k.low, k.close, k.volume)  # 直接取 float

# ── 财务数据 ──
fin = stock.get_finance_info()
roe = float(fin["jinglirun"]) / float(fin["jingzichan"])        # ROE
nav_ps = float(fin["meigujingzichan"])                           # 每股净资产

# ── 交易日历 ──
cal = sm.get_trading_calendar(Query(-30))                        # 最近 30 个交易日

# ── 指数成分股 ──
idx = sm["sz399101"]                                             # 中小板指
blocks = sm.get_block_list_by_index_stock(idx)
stocks = blocks[0].get_stock_list()                              # Stock 对象列表
for stk in stocks:
    code = stk.market_code                                       # "sz002xxx"
    name = stk.name

# ── Datetime 转换 ──
dt = Datetime(2024, 6, 15)                                       # 构造日期
# str / datetime / Datetime 统一转换见 _to_hikyuu_date() 工具函数
```

### 数据适配层

为避免策略代码与 Hikyuu API 过度耦合，项目通过 `trade_advisor/data_adapter.py` 封装了一层统一接口：

```python
from trade_advisor.data_adapter import get_kdata, get_stock_list, get_market_cap

df = get_kdata("sz000001", "2024-01-01", "2024-06-09")  # 返回 pandas DataFrame
stocks = get_stock_list("small_cap")                     # 返回代码列表
cap = get_market_cap("sh600519")                         # 返回流通市值（亿元）
```

该适配层的行为：
- **默认优先尝试 Hikyuu**：数据从本地 HDF5 读取，毫秒级响应
- **失败自动降级**：若 Hikyuu 无数据（如未下载该时段数据），自动调用 AKShare 网络 API 补充
- **策略层无感知**：策略只需调用 `get_kdata()`，无需关心数据来源

### HDF5 数据时效性检测

`trade_advisor/data_status.py` 提供 HDF5 文件的时效性检测：

```python
from trade_advisor.data_status import get_db_latest_dates, get_data_freshness

dates = get_db_latest_dates()
# → {"sh": "2024-06-07", "sz": "2024-06-07", "overall": "2024-06-07"}

status = get_data_freshness()
# → {"is_fresh": True, "days_diff": 2, "status_label": "🟢 数据较新"}
```

---

## 策略总览

### 核心策略（5 个）

| 策略 | 类型 | 说明 |
|------|------|------|
| **小市值策略** | 基本面选股 | 中小板指成分股中选取流通市值最小的 N 只，自动过滤 ST/涨跌停/停牌 |
| **多金叉共振** | 技术形态 | MA5 上穿 MA20 且放量时买入，MA5 下穿 MA20 时卖出 |
| **启明星形态** | 技术形态 | 三根 K 线（长阴→星线→长阳）底部反转识别 |
| **科技动量轮动** | 动量轮动 | 固定科技股池（光模块/芯片/航天），加权对数线性回归动量打分 |
| **ETF 双池动量轮动** | ETF 轮动 | 全市场 ETF 静态+动态池融合，加权平滑动量打分，8% 止损，防御 ETF 自动切换 |

### 新增策略

| 策略 | 说明 |
|------|------|
| **W 底形态识别** | 经典双底反转形态自动扫描 |
| **涨停回调买点** | 涨停后缩量回调至关键均线的二次介入点 |
| **多金叉共振 V2** | 多金叉共振的另一种参数实现 |
| **趋势动量因子** | 基于趋势强度与动量因子的复合评分 |
| **增强多因子** | 三级防御体系 + 趋势/动量/量能/质量/换手五维评分 |
| **ML 因子** | 因子评分可扩展为机器学习模型推理入口 |

---

## 架构

```
stock_advisor/
├── start_trade_advisor.py     # 启动入口（带日志与异常弹窗）
├── requirements.txt           # 依赖声明
│
├── trade_advisor/
│   ├── main.py                # 模块启动入口
│   ├── ui.py                  # PySide6 图形界面
│   ├── data_adapter.py        # 数据适配层（Hikyuu + AKShare 自动降级）
│   ├── backtest.py            # 回测引擎
│   ├── storage.py             # SQLite 持久化
│   ├── data_status.py         # HDF5 数据时效性检测
│   ├── etf_pool.py            # ETF 静态池（130+ 只）
│   ├── ml_predictor.py        # ML 模型加载入口
│   │
│   └── strategies/
│       ├── __init__.py        # 基类 + 策略注册表 + 5 个核心策略
│       ├── w_bottom.py
│       ├── limit_up_pullback.py
│       ├── golden_cross_v2.py
│       ├── trend_momentum.py
│       ├── enhanced_multifactor.py
│       ├── market_risk_monitor.py
│       └── ml_strategy.py
│
├── models/                    # ML 模型存放目录
├── docs/                      # 参考资料
└── .gitignore
```

### 数据流

```
Hikyuu HDF5 ──→ data_adapter.py ──→ BaseStrategy.run() ──→ decisions.db
     │                                      │
     └── AKShare (fallback)                 └── backtest.py → 绩效报告
```

---

## 常见问题

### Hikyuu 数据未更新

运行 `python update_data.py` 从通达信服务器下载最新日线数据到本地 HDF5。

### 策略扫描速度慢

确认 HDF5 文件（`sh_day.h5` / `sz_day.h5`）存在且完整。首次运行或大量策略并发扫描时较慢属正常现象。

### 无法连接通达信服务器

检查网络连接；可在 `update_data.py` 中修改 `TDX_SERVERS` 列表更换服务器地址。

### 如何添加新策略

在 `trade_advisor/strategies/` 中创建新文件，继承 `BaseStrategy`，然后在 `__init__.py` 中注册即可。详见代码注释中的开发指南。
