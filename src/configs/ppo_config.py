from pathlib import Path  
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pydantic import BaseModel
import os, sys
from pathlib import Path


@dataclass
class PPOTrainingConfig:
    """PPO训练配置"""
    base_model_path:str = ""

    reward_model_path:str = ""

    output_dir = ""
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

    cached_train_data_path : str = "/root/autodl-tmp/Travel-Agent-based-on-Qwen2-RLHF/src/data/ppo_data_cached"


@dataclass
class PPOConfig:
    """PPO训练总配置"""
    training: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)
    


