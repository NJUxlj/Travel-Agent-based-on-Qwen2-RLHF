from pathlib import Path  
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pydantic import BaseModel
import os, sys
from pathlib import Path


@dataclass
class ModelConfig:
    """基础模型配置"""
    # 模型名称或路径
    name: str = "/root/autodl-tmp/models/Qwen2.5-0.5B"
    # 模型类型
    model_type: str = "causal_lm"
    # 是否使用混合精度训练
    use_fp16: bool = True
    # 是否使用CPU还是GPU
    device: str = "cuda"  # 或 "cpu"
    # 模型最大输入长度
    max_length: int = 2048
    # 是否使用梯度检查点来节省显存
    gradient_checkpointing: bool = True


