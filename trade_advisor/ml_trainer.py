"""
ML 模型训练工具 — 独立运行，训练完成后策略自动加载

用法：
    conda activate stock
    python -m trade_advisor.ml_trainer              # 默认训练
    python -m trade_advisor.ml_trainer --force       # 强制重新训练

输出文件（保存在 models/ 目录）：
    ml_model.pkl         — 训练好的模型（含三个子模型）
    ml_features.json     — 特征列表
    ml_info.json         — 训练元数据（日期范围、样本数、AUC）

需要安装：
    pip install lightgbm
"""

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_trainer")

DEFAULT_LOOKBACK = 500
DEFAULT_FWD_DAYS = 20
DEFAULT_TRAIN_RATIO = 0.7


# =====================================================================
# 1. 特征工程
# =====================================================================

def compute_features(df: pd.DataFrame, fin_info: dict = None) -> Optional[pd.DataFrame]:
    """
    从 K 线 DataFrame 计算技术因子 + 基本面因子

    Args:
        df: OHLCV DataFrame (按日期升序)
        fin_info: Hikyuu get_finance_info() 返回的字典

    Returns:
        含原始列 + 因子列的 DataFrame
    """
    if df.empty or len(df) < 60:
        return None

    data = df.copy()
    data = data.sort_values("date", ascending=True)
    n = len(data)

    close = data["close"].values.astype(float)
    high = data["high"].values.astype(float)
    low = data["low"].values.astype(float)

    # ════════════════════════════════════════════
    # 收益率因子（多周期）
    # ════════════════════════════════════════════
    data["ret_1d"] = data["close"].pct_change(1)
    data["ret_3d"] = data["close"].pct_change(3)
    data["ret_5d"] = data["close"].pct_change(5)
    data["ret_10d"] = data["close"].pct_change(10)
    data["ret_20d"] = data["close"].pct_change(20)

    # 超额收益 = close / MA20 - 1
    data["excess_ret"] = data["close"] / data["close"].rolling(20).mean() - 1

    # ════════════════════════════════════════════
    # 均线位置因子
    # ════════════════════════════════════════════
    for w in [5, 10, 20, 60]:
        ma = data["close"].rolling(w).mean()
        data[f"close_ma{w}"] = data["close"] / ma - 1

    # 均线间距离
    ma5 = data["close"].rolling(5).mean()
    ma10 = data["close"].rolling(10).mean()
    ma20 = data["close"].rolling(20).mean()
    ma60 = data["close"].rolling(60).mean()

    data["ma5_ma10"] = ma5 / ma10 - 1
    data["ma10_ma20"] = ma10 / ma20 - 1
    data["ma20_ma60"] = ma20 / ma60 - 1

    # 多头排列强度（ma5>ma10>ma20>ma60 各+1分）
    bull = ((ma5 > ma10).astype(int)
            + (ma10 > ma20).astype(int)
            + (ma20 > ma60).astype(int))
    data["bull_strength"] = bull

    # ════════════════════════════════════════════
    # 量能因子
    # ════════════════════════════════════════════
    data["vol_ratio"] = data["volume"] / data["volume"].rolling(5).mean()
    data["vol_ma5_20"] = data["volume"].rolling(5).mean() / data["volume"].rolling(20).mean() - 1
    data["vol_ma20_60"] = data["volume"].rolling(20).mean() / data["volume"].rolling(60).mean() - 1

    # 量价配合
    data["vol_price"] = data["ret_1d"] * data["vol_ratio"]

    # ════════════════════════════════════════════
    # 波动率因子
    # ════════════════════════════════════════════
    data["vol_20d"] = data["ret_1d"].rolling(20).std()
    data["vol_60d"] = data["ret_1d"].rolling(60).std()

    # ════════════════════════════════════════════
    # 价格位置因子
    # ════════════════════════════════════════════
    data["high_20_pos"] = ((data["close"] - data["low"].rolling(20).min())
                           / (data["high"].rolling(20).max() - data["low"].rolling(20).min() + 1e-10))
    data["high_60_pos"] = ((data["close"] - data["low"].rolling(60).min())
                           / (data["high"].rolling(60).max() - data["low"].rolling(60).min() + 1e-10))

    # 均线距离（中长期趋势强度）
    data["ma60_dist"] = data["close"] / ma60 - 1

    # ════════════════════════════════════════════
    # RSI(14)
    # ════════════════════════════════════════════
    delta = data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi_14"] = 100 - 100 / (1 + rs)

    # ════════════════════════════════════════════
    # MACD
    # ════════════════════════════════════════════
    ema12 = data["close"].ewm(span=12).mean()
    ema26 = data["close"].ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    data["macd_dif"] = dif
    data["macd_dea"] = dea
    data["macd_hist"] = (dif - dea) * 2

    # ════════════════════════════════════════════
    # 基本面因子（快照数据，非时序）
    # ════════════════════════════════════════════
    data["log_market_cap"] = np.nan
    data["log_turnover"] = np.nan
    data["log_pe"] = np.nan
    data["log_pb"] = np.nan

    if fin_info:
        ltgb = float(fin_info.get("liutongguben", 0))       # 流通股本（股）
        jlr = float(fin_info.get("jinglirun", 0))            # 净利润
        mjg = float(fin_info.get("meigujingzichan", 0))      # 每股净资产

        if ltgb > 0:
            for i in range(60, n):
                cp = close[i]
                vol = float(data["volume"].iloc[i])
                mv = cp * ltgb                                 # 总市值
                mv_yi = mv / 1e8                                # 亿元
                tr = vol / ltgb * 100                           # 换手率%

                data.loc[data.index[i], "log_market_cap"] = np.log(max(mv_yi, 1))
                data.loc[data.index[i], "log_turnover"] = np.log(max(tr, 0.01))

                if jlr > 0:
                    pe = mv / jlr
                    data.loc[data.index[i], "log_pe"] = np.log(max(min(pe, 200), 3))
                if mjg > 0:
                    pb = cp / mjg
                    data.loc[data.index[i], "log_pb"] = np.log(max(min(pb, 20), 0.3))

    return data


# ── 最终特征列表 ──
FEATURE_COLS = [
    # 收益率（5个）
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "excess_ret",
    # 均线位置（4个）
    "close_ma5", "close_ma10", "close_ma20", "close_ma60",
    # 均线交叉 + 排列强度（4个）
    "ma5_ma10", "ma10_ma20", "ma20_ma60", "bull_strength",
    # 量能（4个）
    "vol_ratio", "vol_ma5_20", "vol_ma20_60", "vol_price",
    # 波动率（2个）
    "vol_20d", "vol_60d",
    # 价格位置 + 趋势强度（3个）
    "high_20_pos", "high_60_pos", "ma60_dist",
    # 技术指标（4个）
    "rsi_14", "macd_dif", "macd_dea", "macd_hist",
    # 基本面（4个）
    "log_market_cap", "log_turnover", "log_pe", "log_pb",
]
# 总计: 30 个特征


# =====================================================================
# 2. 训练数据准备
# =====================================================================

def build_training_data(
    stock_codes: List[str],
    lookback: int = DEFAULT_LOOKBACK,
    fwd_days: int = DEFAULT_FWD_DAYS,
    max_stocks: int = 500,
) -> pd.DataFrame:
    """
    构建训练数据集
    对每只股票：获取K线 → 计算因子 → 生成标签 → 拼接
    """
    from hikyuu.interactive import sm, Query, Days

    all_dfs = []
    processed = 0

    for idx, code in enumerate(stock_codes[:max_stocks]):
        try:
            stk = sm[code]
            q = Query(-lookback - fwd_days - 30)
            k = stk.get_kdata(q)
            if not k or len(k) < lookback:
                continue

            # 转 DataFrame
            recs = [{
                "date": bar.datetime.datetime(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            } for bar in k]
            df = pd.DataFrame(recs).sort_values("date")

            # 获取财务快照
            fin = stk.get_finance_info()
            fin_dict = None
            if fin:
                fin_dict = {k: fin[k] for k in fin.keys()}

            # 计算因子
            feat_df = compute_features(df, fin_dict)
            if feat_df is None or len(feat_df) < lookback:
                continue

            # 标签：未来 fwd_days 收益
            close_vals = feat_df["close"].values
            fwd_ret = np.full(len(close_vals), np.nan)
            for i in range(len(close_vals) - fwd_days):
                fwd_ret[i] = close_vals[i + fwd_days] / close_vals[i] - 1
            feat_df["fwd_ret"] = fwd_ret
            feat_df["code"] = code

            # 丢弃前 60 天（因子 warmup）
            all_dfs.append(feat_df.iloc[60:])
            processed += 1

            if processed % 50 == 0:
                logger.info(f"  进度: {processed}/{min(max_stocks, len(stock_codes))}")

        except Exception:
            continue

    logger.info(f"完成: {processed} 只成功")
    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.dropna(subset=FEATURE_COLS + ["fwd_ret"])
    return result


# =====================================================================
# 3. 模型训练（含验证指标输出）
# =====================================================================

def train_models(train_data: pd.DataFrame):
    import lightgbm as lgb

    train_data = train_data.sort_values("date")
    unique_dates = train_data["date"].unique()
    split_idx = int(len(unique_dates) * DEFAULT_TRAIN_RATIO)
    train_dates = set(unique_dates[:split_idx])
    val_dates = set(unique_dates[split_idx:])

    train_mask = train_data["date"].isin(train_dates)
    val_mask = train_data["date"].isin(val_dates)

    X_train = train_data[train_mask][FEATURE_COLS].values.astype(np.float32)
    X_val = train_data[val_mask][FEATURE_COLS].values.astype(np.float32)

    y_reg = train_data[train_mask]["fwd_ret"].values.astype(np.float32)
    y_reg_val = train_data[val_mask]["fwd_ret"].values.astype(np.float32)

    median_ret = np.median(y_reg)
    y_cls = (y_reg > median_ret).astype(np.int32)
    y_cls_val = (y_reg_val > median_ret).astype(np.int32)

    y_dir = (y_reg > 0).astype(np.int32)
    y_dir_val = (y_reg_val > 0).astype(np.int32)

    logger.info(f"训练集: {len(X_train)}  验证集: {len(X_val)}")
    logger.info(f"特征数: {len(FEATURE_COLS)}")

    base_params = {
        "seed": 42, "verbose": -1,
        "num_leaves": 31, "max_depth": 6,
        "min_child_samples": 30,
        "lambda_l1": 0.2, "lambda_l2": 0.2,
        "learning_rate": 0.05,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "feature_fraction": 0.8,
        "num_threads": 4, "deterministic": True,
    }

    es_cb = lgb.early_stopping(20)
    log_cb = lgb.log_evaluation(0)

    models = {}

    # 分类模型
    logger.info("训练分类模型 (label > median)...")
    m = lgb.train({**base_params, "objective": "binary", "metric": "auc"},
                  lgb.Dataset(X_train, y_cls),
                  valid_sets=[lgb.Dataset(X_val, y_cls_val)],
                  num_boost_round=200, callbacks=[es_cb, log_cb])
    pred = m.predict(X_val)
    auc = _fast_auc(y_cls_val, pred)
    logger.info(f"  → 验证集 AUC: {auc:.4f}")
    models["cls"] = m

    # 回归模型
    logger.info("训练回归模型 (fwd_ret)...")
    m = lgb.train({**base_params, "objective": "regression", "metric": "rmse"},
                  lgb.Dataset(X_train, y_reg),
                  valid_sets=[lgb.Dataset(X_val, y_reg_val)],
                  num_boost_round=200, callbacks=[es_cb, log_cb])
    pred = m.predict(X_val)
    rmse = np.sqrt(np.mean((y_reg_val - pred) ** 2))
    logger.info(f"  → 验证集 RMSE: {rmse:.4f}")
    models["reg"] = m

    # 方向模型
    logger.info("训练方向模型 (fwd_ret > 0)...")
    m = lgb.train({**base_params, "objective": "binary", "metric": "auc"},
                  lgb.Dataset(X_train, y_dir),
                  valid_sets=[lgb.Dataset(X_val, y_dir_val)],
                  num_boost_round=200, callbacks=[es_cb, log_cb])
    pred = m.predict(X_val)
    auc = _fast_auc(y_dir_val, pred)
    logger.info(f"  → 验证集 AUC: {auc:.4f}")
    models["dir"] = m

    # 特征重要性
    logger.info("\n特征重要性（三模型累计 Gain）:")
    imp = {}
    for name, m in models.items():
        fi = m.feature_importance(importance_type="gain")
        for i, f in enumerate(FEATURE_COLS):
            imp[f] = imp.get(f, 0) + fi[i]
    total = sum(imp.values())
    for f, v in sorted(imp.items(), key=lambda x: -x[1]):
        pct = v / total * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  {f:20s} {pct:5.1f}% {bar}")

    return models


def _fast_auc(y_true, y_pred):
    """快速计算 AUC（无需 sklearn）"""
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank = np.argsort(y_pred)
    pos_rank_sum = np.sum(rank[np.argsort(y_true)][-int(n_pos):])
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


# =====================================================================
# 4. 模型持久化
# =====================================================================

MODELS_DIR = PROJECT_ROOT / "models"


def save_models(models: Dict, feature_names: List[str], train_date: str,
                n_samples: int, stock_count: int, metrics: dict = None):
    MODELS_DIR.mkdir(exist_ok=True)
    package = {
        "models": models,
        "feature_names": feature_names,
        "train_date": train_date,
        "n_samples": n_samples,
        "version": 1,
    }
    path = MODELS_DIR / "ml_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(package, f)

    meta = {
        "train_date": train_date,
        "n_samples": n_samples,
        "stock_count": stock_count,
        "feature_count": len(feature_names),
        "metrics": metrics or {},
        "version": 1,
    }
    with open(MODELS_DIR / "ml_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"模型已保存: {path} ({os.path.getsize(path)/1024:.0f} KB)")


def load_models() -> Optional[Dict]:
    path = MODELS_DIR / "ml_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_score(model_package: Dict, df: pd.DataFrame, fin_info: dict = None) -> float:
    """对一只股票预测综合评分 (0~100)"""
    models = model_package["models"]
    feat_df = compute_features(df, fin_info)
    if feat_df is None or len(feat_df) < 2:
        return 0
    X = feat_df.iloc[-1:][model_package["feature_names"]].values.astype(np.float32)
    if np.any(np.isnan(X)):
        return 0
    pred_cls = models["cls"].predict(X)[0]
    pred_reg = models["reg"].predict(X)[0]
    pred_dir = models["dir"].predict(X)[0]
    score = pred_cls * 40 + pred_dir * 30 + max(0, pred_reg / 0.1) * 30
    return float(max(0, min(100, score)))


# =====================================================================
# 5. 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="ML 模型训练工具")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--fwd-days", type=int, default=DEFAULT_FWD_DAYS)
    parser.add_argument("--max-stocks", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.force and (MODELS_DIR / "ml_model.pkl").exists():
        logger.info("模型已存在（使用 --force 重新训练）")
        if (MODELS_DIR / "ml_info.json").exists():
            with open(MODELS_DIR / "ml_info.json") as f:
                print(json.dumps(json.load(f), indent=2))
        return

    logger.info("=" * 50)
    logger.info("ML 模型训练")
    logger.info(f"回看: {args.lookback}天, 未来窗口: {args.fwd_days}天, 股票: {args.max_stocks}只")
    logger.info("=" * 50)

    from hikyuu.interactive import sm
    idx = sm["sz399101"]
    blk_list = sm.get_block_list_by_index_stock(idx)
    if not blk_list:
        logger.error("无法获取股票池")
        return
    stocks = blk_list[0].get_stock_list()
    codes = [s.market_code.lower() for s in stocks]
    logger.info(f"股票池: {len(codes)} 只")

    # 构建数据
    logger.info("\n[1/3] 构建训练数据...")
    train_data = build_training_data(codes, args.lookback, args.fwd_days, args.max_stocks)
    if train_data.empty:
        logger.error("训练数据为空")
        return
    logger.info(f"样本: {len(train_data)}, 特征: {len(FEATURE_COLS)}, "
                f"股票: {train_data['code'].nunique()}只")
    logger.info(f"日期: {train_data['date'].min().date()} ~ {train_data['date'].max().date()}")

    # 预处理：去极值 + 标准化
    logger.info("\n[1b/3] 预处理因子...")
    for col in FEATURE_COLS:
        # 去极值（MAD）
        med = train_data[col].median()
        mad = (train_data[col] - med).abs().median() * 1.4826
        if mad > 0:
            train_data[col] = train_data[col].clip(med - 3*mad, med + 3*mad)
        # 标准化
        mean_v = train_data[col].mean()
        std_v = train_data[col].std()
        if std_v > 0:
            train_data[col] = (train_data[col] - mean_v) / std_v

    # 训练
    logger.info("\n[2/3] 训练模型...")
    models = train_models(train_data)

    # 保存
    logger.info("\n[3/3] 保存模型...")
    metrics = {}
    save_models(models, FEATURE_COLS, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(train_data), train_data["code"].nunique(), metrics)

    logger.info("\n✅ 训练完成！启动系统后选择「ML 因子策略」即可使用")


if __name__ == "__main__":
    main()
