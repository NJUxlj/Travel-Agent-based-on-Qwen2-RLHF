from pathlib import Path  
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pydantic import BaseModel
import os, sys
from pathlib import Path


@dataclass
class GRPOTrainingConfig:
    """GRPO训练配置"""
    # 基础模型路径
    base_model_path: str = "/root/autodl-tmp/models/Qwen2.5-0.5B"

    output_dir: str = "output/"

    # 批量大小
    batch_size: int = 8
    # 进程数
    num_processes: int = 2
    # 学习率
    learning_rate: float = 1e-5

    weight_decay: float = 0.05  # 权重衰减
    max_grad_norm: float = 1.0  # 最大梯度范数
    # 训练轮数
    num_train_epochs: int = 3
    # 梯度累积步数
    gradient_accumulation_steps: int = 4
    # 日志记录步数
    logging_steps: int = 10
    # 保存步数
    save_steps: int = 100

    clip_upper_bound: float = 0.2
    clip_lower_bound: float = -0.2
    
    # GRPO特有参数
    mini_batch_size: int = 4  # 小批量大小
    num_groups: int = 4  # 组数

    # KL惩罚参数
    use_kl_loss: bool = False  # 是否使用KL损失
    kl_loss_coef: float = 0.001  # KL损失系数
    kl_loss_type: str = "low_var_kl"  # KL损失类型
    kl_penalty: float = 0.1  # KL惩罚系数
    
    # 训练稳定性参数
    entropy_coeff: float = 0.0  # 熵系数，增加探索性
    ppo_epochs: int = 1  # GRPO更新轮数
    
    # 生成配置
    prompt_truncation_side: str = "right"  # 提示截断方向
    
    # DrGRPO相关配置（可选）
    use_drgrpo: bool = False  # 是否使用DrGRPO变体
    loss_agg_mode: str = "seq-mean-token-sum-norm"  # 损失聚合模式
    norm_adv_by_std: bool = False  # 是否通过标准差标准化优势

    # 数据路径配置
    cached_train_data_path: str = "/root/autodl-tmp/Travel-Agent-based-on-Qwen2-RLHF/src/data/grpo_data_cached"




@dataclass
class GRPOConfig:
    """GRPO训练总配置"""
    training: GRPOTrainingConfig = field(default_factory=GRPOTrainingConfig)


    # 后期这里可以挂载更多的子配置类