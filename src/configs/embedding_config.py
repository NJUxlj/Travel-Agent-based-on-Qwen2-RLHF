from pathlib import Path  
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pydantic import BaseModel
import os, sys
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    # 嵌入模型路径
    embedding_model_path: str = "/root/autodl-tmp/models/all-MiniLM-L6-v2"

