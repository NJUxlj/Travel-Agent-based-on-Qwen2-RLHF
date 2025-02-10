from dataclasses import dataclass
from typing import Dict, List, Optional
from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer

@dataclass
class DPOConfig:
    """DPO训练配置"""
    beta: float = 0.1
    max_prompt_length: int = 512
    max_length: int = 1024
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    
class DPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        config: DPOConfig = DPOConfig(),
        output_dir: str = "./dpo_results"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # 设置训练参数
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=config.gradient_checkpointing,
            remove_unused_columns=False,
        )
        
    def prepare_dataset(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str]
    ) -> Dataset:
        """准备DPO训练数据集"""
        dataset_dict = {
            "prompt": prompts,
            "chosen": chosen_responses,
            "rejected": rejected_responses,
        }
        return Dataset.from_dict(dataset_dict)
    
    def train(self, dataset: Dataset):
        """执行DPO训练"""
        trainer = DPOTrainer(
            model=self.model,
            args=self.training_args,
            beta=self.config.beta,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.max_prompt_length,
            max_length=self.config.max_length,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model()