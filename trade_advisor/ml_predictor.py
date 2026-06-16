"""
ML 模型预测器 — 被策略加载使用

加载已训练的模型并对股票进行评分。

用法：
    from trade_advisor.ml_trainer import load_models, predict_score

    model = load_models()
    score = predict_score(model, df_kline)
"""

from pathlib import Path
from typing import Dict, Optional
import os
import pickle
import json


MODELS_DIR = Path(__file__).parent.parent / "models"


def get_model_info() -> Optional[Dict]:
    """获取已训练模型的元数据"""
    info_path = MODELS_DIR / "ml_info.json"
    if not info_path.exists():
        return None
    with open(info_path, "r") as f:
        return json.load(f)


def model_available() -> bool:
    """检查模型文件是否存在"""
    return (MODELS_DIR / "ml_model.pkl").exists()
