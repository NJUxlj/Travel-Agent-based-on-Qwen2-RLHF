from pathlib import Path  
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pydantic import BaseModel
import os, sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union


@dataclass
class CPTTrainingConfig:
    """继续预训练训练配置类"""
    # 学习率配置
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 训练轮数和批次大小
    num_train_epochs: int = 3
    max_steps: int = -1  # 如果为-1，则使用num_train_epochs
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"  # linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    warmup_ratio: float = 0.05
    warmup_steps: int = 0
    
    # 保存和日志配置
    output_dir: str = "./cpt_output"
    logging_dir: str = "./cpt_output/logs"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    save_strategy: str = "steps"  # steps, epoch, no
    evaluation_strategy: str = "steps"  # steps, epoch, no
    
    # 优化器配置
    optim: str = "adamw_torch"  # adamw_hf, adamw_torch, adafactor, muon
    optim_args: Optional[str] = None
    
    # 早停配置
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # 其他训练配置
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    fp16_opt_level: str = "O1"
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 梯度检查点和内存优化
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    
    # 分布式训练配置
    ddp_backend: Optional[str] = None
    ddp_broadcast_buffers: bool = False
    ddp_find_unused_parameters: Optional[bool] = None
    ddp_timeout: int = 1800
    
    # 其他高级配置
    local_rank: int = -1
    tpu_num_cores: Optional[int] = None
    dataloader_drop_last: bool = True
    eval_delay: int = 0


@dataclass
class CPTModelConfig:
    """继续预训练模型配置类"""
    # 模型路径和配置
    model_name_or_path: str = "Qwen/Qwen2.5-7B"
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    use_fast_tokenizer: bool = True
    model_revision: str = "main"
    token: Optional[str] = None
    trust_remote_code: bool = True
    
    # 模型加载配置
    torch_dtype: Optional[str] = "bfloat16"  # float32, float16, bfloat16
    low_cpu_mem_usage: bool = True
    device_map: Optional[str] = "auto"  # auto, balanced, max_memory
    
    # 序列长度配置
    max_length: int = 2048
    truncation: bool = True
    padding: str = "max_length"
    
    # 注意力机制配置
    use_flash_attention: bool = True
    use_cache: bool = True
    
    # 特殊令牌配置
    pad_token: Optional[str] = None
    eos_token: Optional[str] = None
    bos_token: Optional[str] = None
    unk_token: Optional[str] = None
    
    # 其他模型配置
    use_auth_token: Optional[str] = None
    resize_token_embeddings: bool = True


@dataclass
class CPTDataConfig:
    """继续预训练数据配置类"""
    # 数据集配置
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    data_files: Optional[List[str]] = None
    
    # 数据处理配置
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_predict_samples: Optional[int] = None
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    
    # 数据预处理配置
    preprocessing_num_workers: Optional[int] = None
    overwrite_cache: bool = False
    mlm_probability: float = 0.15  # 仅用于MLM训练
    line_by_line: bool = False
    
    # 数据格式配置
    text_column: str = "text"
    ignore_pad_token_for_loss: bool = True
    
    # 数据增强配置
    use_data_augmentation: bool = False
    augmentation_prob: float = 0.1
    
    # 数据过滤配置
    min_length: int = 10
    max_length_percentile: int = 95
    filter_duplicates: bool = True


@dataclass
class CPTConfig:
    """继续预训练总配置类，整合所有配置"""
    # 子配置
    training: CPTTrainingConfig = field(default_factory=CPTTrainingConfig)
    model: CPTModelConfig = field(default_factory=CPTModelConfig)
    data: CPTDataConfig = field(default_factory=CPTDataConfig)
    
    # 实验配置
    experiment_name: str = "cpt_experiment"
    run_name: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # 其他配置
    resume_from_checkpoint: Optional[str] = None
    sortish_sampler: bool = False
    predict_with_generate: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CPTConfig":
        """从字典创建配置"""
        # 提取子配置
        training_config = config_dict.get("training", {})
        model_config = config_dict.get("model", {})
        data_config = config_dict.get("data", {})
        
        # 创建配置实例
        training = CPTTrainingConfig(**training_config)
        model = CPTModelConfig(**model_config)
        data = CPTDataConfig(**data_config)
        
        # 创建主配置
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ["training", "model", "data"]}
        
        return cls(
            training=training,
            model=model,
            data=data,
            **main_config
        )
    
    def save_to_file(self, file_path: str):
        """将配置保存到文件"""
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            import json
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> "CPTConfig":
        """从文件加载配置"""
        with open(file_path, "r", encoding="utf-8") as f:
            import json
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> "CPTConfig":
        """从文件加载配置（load_from_file的别名）"""
        return cls.load_from_file(file_path)