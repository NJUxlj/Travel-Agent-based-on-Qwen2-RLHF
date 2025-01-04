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
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel


import sys
sys.path.append("../../")  # 添加上级目录的上级目录到sys.path
sys.path.append("../")
from configs.config import MODEL_CONFIG, BATCH_SIZE
from models.model import TravelAgent
from utils import (
    parse_args,
    get_max_length_from_model,
    check_deepspeed_env,
    check_deepspeed_config,
    load_qwen_in_4bit,
    SFTArguments,
    monitor_memory
)

from data.data_processor import DataProcessor, CrossWOZProcessor
from contextlib import contextmanager


MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-1.5B"


'''
python sft_trainer.py \
--model_name "/root/autodl-tmp/models/Qwen2.5-1.5B" \
--output_dir "output" \
--device "cuda" \
--device_map "auto"


deepspeed --num_gpus=2 sft_trainer.py \
--deepspeed ds_config.json \
--model_name "/root/autodl-tmp/models/Qwen2.5-1.5B" \
--output_dir "output" \
--device "cuda" \
--device_map "auto"

deepspeed --num_gpus=2 sft_trainer.py \
--deepspeed ds_config.json


deepspeed --num_gpus 2 sft_trainer.py \
    --deepspeed ds_config.json

'''

class CustomTrainer(Trainer):  
    @contextmanager  
    def compute_loss_context_manager(self):  
        """  
        重写这个方法以禁用 no_sync 上下文管理器  
        """  
        if self.args.gradient_accumulation_steps > 1:  
            if self.deepspeed:  
                # 对于 deepspeed，我们直接返回一个空的上下文管理器  
                yield  
            else:  
                # 对于非 deepspeed，保持原有行为  
                if self.model.is_gradient_checkpointing:  
                    # 如果使用了梯度检查点，不要使用 no_sync  
                    yield  
                else:  
                    with self.model.no_sync():  
                        yield  
        else:  
            yield  

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
        if check_deepspeed_env():
            pass
        else:
            raise ValueError("DeepSpeed is not installed or not configured correctly.")
        
        
        self.model_name = args.model_name
        self.output_dir = args.output_dir
        self.device = args.device
        self.device_map = args.device_map
        self.local_rank = args.local_rank
        
        if self.local_rank!=-1:
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = args.device
        
        # 加载模型和分词器 # 添加LoRA
        self.agent=TravelAgent(
            model_name=self.model_name,
            device=self.device,
            device_map=self.device_map,
            lora_config=lora_config,
            use_bnb=True
        )
        
        self.model = self.agent.model
        self.max_length = get_max_length_from_model(self.model)
        self.tokenizer = self.agent.tokenizer
        
        '''
        无论选择哪种方案，确保：
            DeepSpeed的train_batch_size等于实际的总batch size
            DeepSpeed的train_micro_batch_size_per_gpu与TrainingArguments的per_device_train_batch_size相等
            所有数值满足：total_batch = micro_batch * num_gpus * grad_accum
        '''

        # 设置默认训练参数
        default_training_args = TrainingArguments(  
            output_dir=self.output_dir,  
            num_train_epochs=3,  
            per_device_train_batch_size=1,  # 每个GPU上的batch size
            per_device_eval_batch_size=1,  
            gradient_accumulation_steps=4,  
            learning_rate=2e-4,  
            weight_decay=0.01,  
            warmup_steps=100,
            warmup_ratio=0.03,  
            lr_scheduler_type="cosine",  
            # 改用 bf16 而不是 fp16，因为 bf16 数值稳定性更好  
            bf16=True,  # 修改这里  
            fp16=False, # 关闭 fp16 
            # fp16=True,  
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
            # 添加以下参数来启用 8-bit 优化器  
            optim="paged_adamw_8bit",  
        )  
        
        # 更新训练参数
        if training_args:
            self.training_args = training_args
        else:
            self.training_args = default_training_args
        
        check_deepspeed_config(self.training_args)
    

    
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
        data_collator = DataCollatorForSeq2Seq(  
            tokenizer=self.tokenizer,  
            model=self.model,
            max_length=1024, # self.max_length,
            padding="max_length",
            return_tensors="pt",
            # mlm=False  
        )  
        
        # 创建训练器
        trainer = CustomTrainer(
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
        
        monitor_memory()
        # 开始训练
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 只在主进程保存模型  
        if args.local_rank in [-1, 0]:  
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
    args = SFTArguments()  # 使用parse_args获取参数
    trainer = SFTTrainer(args = args)
    
    processor = CrossWOZProcessor(
        tokenizer=trainer.tokenizer,
        max_length = trainer.max_length,
        system_prompt=None  
    )
    
    
    data_path = "/root/autodl-tmp/Travel-Agent-based-on-LLM-and-SFT/data/processed/crosswoz_sft"
    processed_data = processor.process_conversation_data_huggingface(data_path)
    
    
    trainer.train(
        train_dataset=processed_data["train"],
        eval_dataset=processed_data["test"]
    )
    
    
