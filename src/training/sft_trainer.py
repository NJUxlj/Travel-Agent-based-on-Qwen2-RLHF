from typing import Dict, Optional
import os
import torch
import evaluate
import deepspeed
import transformers
from transformers import (
    Trainer, 
    TrainingArguments,
    default_data_collator,
    DataCollatorForLanguageModeling,  
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel


import sys
sys.path.append("../../")  # 添加上级目录的上级目录到sys.path
sys.path.append("../")
from configs.config import MODEL_CONFIG
from models.model import TravelAgent
from utils import (
    parse_args,
    get_max_length_from_model,
)

from data.data_processor import DataProcessor



MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-1.5B"


'''
python sft_trainer.py \
--model_name "/root/autodl-tmp/models/Qwen2.5-1.5B" \
--output_dir "output" \
--device "cuda" \
--device_map "auto"


'''

class SFTTrainer:
    """
    监督微调训练器
    """
    def __init__(
        self,
        # model_name: str,
        # output_dir: str,
        training_args: Optional[TrainingArguments] = None,
        # device = "auto",
        # device_map = 'auto',
        lora_config: Optional[Dict] = None,
        args = None
    ):
        """
        初始化训练器
        
        Args:
            model_name: 基础模型名称
            output_dir: 输出目录
            training_args: 训练参数
        """
        
        self.model_name = args.model_name
        self.output_dir = args.output_dir
        self.device = args.device
        self.device_map = args.device_map
        
        # 加载模型和分词器 # 添加LoRA
        self.agent=TravelAgent(
            model_name=self.model_name,
            device=self.device,
            device_map=self.device_map,
            lora_config=lora_config
        )
        
        
        self.model = self.agent.model
        self.max_length = get_max_length_from_model(self.model)
        self.tokenizer = self.agent.tokenizer
        

        # 设置默认训练参数
        default_training_args = TrainingArguments(  
            output_dir=self.output_dir,  
            num_train_epochs=3,  
            per_device_train_batch_size=2,  
            per_device_eval_batch_size=2,  
            gradient_accumulation_steps=8,  
            learning_rate=2e-4,  
            weight_decay=0.01,  
            warmup_ratio=0.03,  
            lr_scheduler_type="cosine",  
            fp16=True,  
            logging_steps=100,  
            save_steps=100,  
            eval_steps=100,  
            save_total_limit=3,  
            evaluation_strategy="steps",  
            load_best_model_at_end=True,  
            report_to="tensorboard",  
            # DeepSpeed配置  
            deepspeed="ds_config.json",  
            # 分布式训练配置  
            local_rank=int(os.getenv("LOCAL_RANK", -1)),  
            ddp_find_unused_parameters=False,  
        )  
        
        # 更新训练参数
        if training_args:
            self.training_args = training_args
        else:
            self.training_args = default_training_args
    

    
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
        
        # 数据整理器  
        data_collator = DataCollatorForLanguageModeling(  
            tokenizer=self.tokenizer,  
            mlm=False  
        )  
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[  
                transformers.EarlyStoppingCallback(  
                    early_stopping_patience=3,  
                    early_stopping_threshold=0.01  
                )  
            ]  
        )
        
        # 开始训练
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer
    
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
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        model = AutoModelForCausalLM.from_pretrained(  
            base_model_name,  
            trust_remote_code=True,
            torch_dtype=torch.float16,  
            device_map=device_map
        ) 
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map=device_map
        )
        
        return model, tokenizer
    
    def compute_metrics(self, eval_pred):  
        # 计算评估指标  
        metric = evaluate.load("perplexity")  
        
        predictions, labels = eval_pred  
        # 去除padding的影响  
        mask = labels != -100  
        predictions = predictions[mask]  
        labels = labels[mask]  
        
        return metric.compute(predictions=predictions, references=labels)  
    
    
    


if __name__ == "__main__":
    args = parse_args()  # 使用parse_args获取参数
    trainer = SFTTrainer(args)
    
    processor = DataProcessor(
        tokenizer=trainer.tokenizer,
        max_length = trainer.max_length,
        system_prompt=None  
    )
    
    processor.process
    # trainer.train()