from pathlib import Path  
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pydantic import BaseModel
import os, sys
from pathlib import Path


@dataclass
class DPOTrainingConfig:
    """DPO训练基础参数"""
    base_model_path:str = "/root/autodl-tmp/models/Qwen2.5-0.5B"
    # 输出目录
    output_dir: str = "./dpo_outputs"
    # DPO beta参数（控制KL惩罚强度）
    beta: float = 0.1
    # 训练轮数
    num_train_epochs: int = 3
    # 每个设备的批次大小
    per_device_train_batch_size: int = 2
    # 梯度累积步数
    gradient_accumulation_steps: int = 8
    # 学习率
    learning_rate: float = 1e-5
    # 是否使用梯度检查点
    gradient_checkpointing: bool = True
    # 评估步数
    eval_steps: int = 100
    # 保存步数
    save_steps: int = 100
    # 日志记录步数
    logging_steps: int = 10
    # 最大保存检查点数量
    save_total_limit: int = 3

    """序列长度配置"""
    # 最大提示长度
    max_prompt_length: int = 512
    # 最大序列总长度
    max_length: int = 1024
    # 是否填充到最大长度
    pad_to_max_length: bool = True

    """优化器配置"""
    # 优化器类型
    optimizer_type: str = "adamw"
    # 权重衰减
    weight_decay: float = 0.01
    # 梯度裁剪
    max_grad_norm: float = 1.0
    # 学习率调度器
    lr_scheduler_type: str = "cosine"
    # 预热步数比例
    warmup_ratio: float = 0.1


@dataclass
class DPODataConfig:
    """数据配置"""
    # 训练数据路径
    train_file: str = "data/processed/dpo_train.json"
    # 验证数据路径
    validation_file: str = "data/processed/dpo_validation.json"
    # 数据格式
    format: str = "json"
    # 提示词列名
    prompt_column: str = "prompt"
    # 选中回复列名
    chosen_column: str = "chosen"
    # 拒绝回复列名
    rejected_column: str = "rejected"



    


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估指标
    metrics: list = field(default_factory=lambda: ["accuracy", "preference_score"])
    # 评估间隔（步数）
    eval_steps: int = 100
    # 是否在训练结束时评估
    evaluate_at_end: bool = True


@dataclass
class DPOConfig:
    """DPO训练总配置"""
    training: DPOTrainingConfig = field(default_factory=DPOTrainingConfig)
    data: DPODataConfig = field(default_factory=DPODataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


