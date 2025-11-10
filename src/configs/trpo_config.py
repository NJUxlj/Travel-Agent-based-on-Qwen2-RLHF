from pathlib import Path  
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pydantic import BaseModel
import os, sys
from pathlib import Path


@dataclass
class TRPOTrainingConfig:
    """TRPO训练配置"""
    base_model_path: str = "/root/autodl-tmp/models/Qwen2.5-0.5B"
    # 奖励模型路径
    reward_model_path: str = "/root/autodl-tmp/models/reward-model-deberta-v3-large-v2"

    output_dir:str = "./output"
    # 批量大小
    batch_size: int = 8
    # 进程数
    num_processes: int = 2
    # 学习率
    learning_rate: float = 1e-5
    # 训练轮数
    num_train_epochs: int = 3
    # 梯度累积步数
    gradient_accumulation_steps: int = 4
    # 日志记录步数
    logging_steps: int = 10
    # 保存步数
    save_steps: int = 100



@dataclass
class TRPOConfig:
    """TRPO训练总配置"""
    training: TRPOTrainingConfig = field(default_factory=TRPOTrainingConfig)


