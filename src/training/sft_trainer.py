from typing import Dict, Optional
import os
import torch
from transformers import Trainer, TrainingArguments
from peft import PeftModel
from ..models.model_utils import ModelUtils

class SFTTrainer:
    """
    监督微调训练器
    """
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        training_args: Optional[Dict] = None
    ):
        """
        初始化训练器
        
        Args:
            model_name: 基础模型名称
            output_dir: 输出目录
            training_args: 训练参数
        """
        self.model_name = model_name
        self.output_dir = output_dir
        
        # 加载模型和分词器
        self.model, self.tokenizer = ModelUtils.load_base_model(model_name)
        
        # 添加LoRA
        self.model = ModelUtils.prepare_model_for_lora(self.model)
        
        # 设置默认训练参数
        default_training_args = {
            "output_dir": output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 3,
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "fp16": True,
            "remove_unused_columns": False,
        }
        
        # 更新训练参数
        if training_args:
            default_training_args.update(training_args)
        
        # 创建训练参数对象
        self.training_args = TrainingArguments(**default_training_args)
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        开始训练
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            resume_from_checkpoint: 恢复训练的检查点路径
        """
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
    
    @staticmethod
    def load_trained_model(
        base_model_name: str,
        adapter_path: str,
        device_map: str = "auto"
    ) -> tuple:
        """
        加载训练好的模型
        
        Args:
            base_model_name: 基础模型名称
            adapter_path: LoRA权重路径
            device_map: 设备映射策略
        
        Returns:
            tuple: (model, tokenizer)
        """
        # 加载基础模型和分词器
        model, tokenizer = ModelUtils.load_base_model(
            base_model_name,
            device_map=device_map
        )
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map=device_map
        )
        
        return model, tokenizer