"""
ETF 静态池 + 分类词典 — ETF双池平滑动量轮动 策略数据层
"""
from dataclasses import dataclass


@dataclass
class ETFInfo:
    """ETF 元信息"""
    code: str          # hikyuu market code, 如 "sh510050"
    name: str          # 中文名称
    category: str      # 分类


# ── 防御 ETF（货币基金，作为空仓替代） ──
DEFENSIVE_ETF_CODE = "SH511880"

# ── 静态核心池（130+ 只主要ETF） ──
STATIC_POOL = [
    # === 宽基 ===
    ETFInfo('SH510050', "上证50ETF", "宽基"),
    ETFInfo('SH510300', "沪深300ETF", "宽基"),
    ETFInfo('SH510310', "HS300ETF", "宽基"),
    ETFInfo('SH510500', "中证500ETF", "宽基"),
    ETFInfo('SH510580', "中证500ETF", "宽基"),
    ETFInfo('SH510880', "红利ETF", "宽基"),
    ETFInfo('SH512100', "中证1000ETF", "宽基"),
    ETFInfo('SH512500', "中证500ETF", "宽基"),
    ETFInfo('SH515180', "中证红利", "宽基"),
    ETFInfo('SH560010', "中证1000ETF", "宽基"),
    ETFInfo('SH588000', "科创50ETF", "宽基"),
    ETFInfo('SH588050', "科创ETF", "宽基"),
    ETFInfo('SH588090', "科创板ETF", "宽基"),
    ETFInfo('SZ159845', "中证1000ETF", "宽基"),
    ETFInfo('SZ159915', "创业板ETF", "宽基"),
    ETFInfo('SZ159919', "沪深300ETF", "宽基"),
    ETFInfo('SZ159922', "中证500ETF", "宽基"),
    ETFInfo('SZ159949', "创业板50", "宽基"),
    ETFInfo('SZ159901', "深证100ETF", "宽基"),
    ETFInfo('SH510210', "综指ETF", "宽基"),
    ETFInfo('SH563000', "中证A50", "宽基"),
    ETFInfo('SZ159601', "A50ETF", "宽基"),
    ETFInfo('SH560300', "中证2000ETF", "宽基"),

    # === 科技半导体 ===
    ETFInfo('SH512480', "半导体ETF", "科技"),
    ETFInfo('SH512760', "芯片ETF", "科技"),
    ETFInfo('SH515000', "科技ETF", "科技"),
    ETFInfo('SH515050', "5GETF", "科技"),
    ETFInfo('SH515880', "通信ETF", "科技"),
    ETFInfo('SH517050', "互联ETF", "科技"),
    ETFInfo('SH560170', "科技50", "科技"),
    ETFInfo('SH561010', "软件ETF", "科技"),
    ETFInfo('SH561100', "云计算ETF", "科技"),
    ETFInfo('SH562500', "机器人ETF", "科技"),
    ETFInfo('SH588300', "双创50ETF", "科技"),
    ETFInfo('SH517010', "数字ETF", "科技"),
    ETFInfo('SZ159995', "芯片ETF", "科技"),
    ETFInfo('SZ159997', "半导体ETF", "科技"),
    ETFInfo('SZ159939', "信息技术", "科技"),
    ETFInfo('SZ159869', "游戏ETF", "科技"),
    ETFInfo('SH560800', "数字经济", "科技"),

    # === 医药 ===
    ETFInfo('SH512010', "医药ETF", "医药"),
    ETFInfo('SH512170', "医疗ETF", "医药"),
    ETFInfo('SH515120', "医药ETF", "医药"),
    ETFInfo('SH560660', "医药健康", "医药"),
    ETFInfo('SH517390', "中药ETF", "医药"),
    ETFInfo('SH560080', "中药ETF", "医药"),
    ETFInfo('SZ159929', "医药ETF", "医药"),
    ETFInfo('SZ159837', "生物医药", "医药"),
    ETFInfo('SZ159828', "医疗ETF", "医药"),
    ETFInfo('SZ159883', "医疗器械", "医药"),
    ETFInfo('SH510850', "生物科技", "医药"),

    # === 消费 ===
    ETFInfo('SH510630', "消费ETF", "消费"),
    ETFInfo('SH517880', "品牌消费", "消费"),
    ETFInfo('SZ159928', "消费ETF", "消费"),
    ETFInfo('SZ159936', "可选消费", "消费"),
    ETFInfo('SH562900', "农业ETF", "消费"),
    ETFInfo('SZ159825', "农业ETF", "消费"),
    ETFInfo('SZ159865', "养殖ETF", "消费"),
    ETFInfo('SZ159766', "旅游ETF", "消费"),
    ETFInfo('SH510150', "消费ETF", "消费"),

    # === 金融 ===
    ETFInfo('SH510230', "金融ETF", "金融"),
    ETFInfo('SH512000', "券商ETF", "金融"),
    ETFInfo('SH512070', "证券保险", "金融"),
    ETFInfo('SH512880', "证券ETF", "金融"),
    ETFInfo('SH512900', "证券ETF", "金融"),
    ETFInfo('SH512800', "银行ETF", "金融"),
    ETFInfo('SZ159940', "金融ETF", "金融"),
    ETFInfo('SZ159993', "金融科技", "金融"),
    ETFInfo('SH513090', "香港证券", "金融"),

    # === 新能源 ===
    ETFInfo('SH515030', "新能源车ETF", "新能源"),
    ETFInfo('SH515790', "光伏ETF", "新能源"),
    ETFInfo('SH516160', "新能源ETF", "新能源"),
    ETFInfo('SH560550', "碳中和ETF", "新能源"),
    ETFInfo('SH561160', "新能源ETF", "新能源"),
    ETFInfo('SH561190', "双碳ETF", "新能源"),
    ETFInfo('SZ159857', "光伏ETF", "新能源"),
    ETFInfo('SZ159637', "新能源车ETF", "新能源"),
    ETFInfo('SH560980', "光伏ETF", "新能源"),

    # === 军工 ===
    ETFInfo('SH512660', "军工ETF", "军工"),
    ETFInfo('SH512670', "国防ETF", "军工"),
    ETFInfo('SH512680', "军工ETF", "军工"),
    ETFInfo('SH512710', "军工龙头", "军工"),
    ETFInfo('SH515010', "军工ETF", "军工"),
    ETFInfo('SZ159656', "军工ETF", "军工"),

    # === 周期 ===
    ETFInfo('SH515220', "煤炭ETF", "周期"),
    ETFInfo('SH516970', "基建ETF", "周期"),
    ETFInfo('SH512400', "有色金属", "周期"),
    ETFInfo('SH515210', "钢铁ETF", "周期"),
    ETFInfo('SH561500', "机床ETF", "周期"),
    ETFInfo('SH562800', "稀有金属", "周期"),
    ETFInfo('SZ159611', "电力ETF", "周期"),
    ETFInfo('SH562300', "低碳ETF", "周期"),
    ETFInfo('SH517090', "央企共赢", "周期"),

    # === 地产 ===
    ETFInfo('SH512200', "房地产ETF", "地产"),
    ETFInfo('SZ159707', "地产ETF", "地产"),

    # === 跨境 ===
    ETFInfo('SH513050', "中概互联", "跨境"),
    ETFInfo('SH513060', "恒生医药", "跨境"),
    ETFInfo('SH513100', "纳指ETF", "跨境"),
    ETFInfo('SH513130', "恒生科技", "跨境"),
    ETFInfo('SH513180', "恒生科技", "跨境"),
    ETFInfo('SH513500', "标普500", "跨境"),
    ETFInfo('SH513520', "日经ETF", "跨境"),
    ETFInfo('SH513800', "东南亚ETF", "跨境"),
    ETFInfo('SZ159605', "中概互联", "跨境"),
    ETFInfo('SZ159607', "中概互联", "跨境"),
    ETFInfo('SZ159920', "恒生ETF", "跨境"),
    ETFInfo('SZ159941', "纳指ETF", "跨境"),
    ETFInfo('SZ159850', "中概互联", "跨境"),
    ETFInfo('SH517100', "港股通ETF", "跨境"),

    # === 商品 ===
    ETFInfo('SH518800', "黄金ETF", "商品"),
    ETFInfo('SH518880', "黄金ETF", "商品"),
    ETFInfo('SZ159934', "黄金ETF", "商品"),
    ETFInfo('SZ159980', "有色ETF", "商品"),
    ETFInfo('SZ159981', "能源化工", "商品"),

    # === 货币（防御） ===
    ETFInfo(DEFENSIVE_ETF_CODE, "银华日利", "货币"),
]


# ── 查找表 ──
CODE_TO_ETF: dict[str, ETFInfo] = {e.code: e for e in STATIC_POOL}

# code → category 映射
CATEGORY_MAP: dict[str, str] = {e.code: e.category for e in STATIC_POOL}

# 分类名称列表
CATEGORIES: list[str] = list(dict.fromkeys(e.category for e in STATIC_POOL))


def get_etf_category(code: str) -> str:
    """获取 ETF 分类，未知返回 '未知'"""
    return CATEGORY_MAP.get(code, "未知")


def get_etf_name(code: str) -> str:
    """获取 ETF 显示名称"""
    etf = CODE_TO_ETF.get(code)
    return etf.name if etf else code
