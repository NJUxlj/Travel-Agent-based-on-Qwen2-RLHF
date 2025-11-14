# 导入一些必要的包
import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Any, Dict, Optional, Union
from dataclasses import dataclass
import logging
from tqdm import tqdm
import math

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 导入配置类
from pathlib import Path
import sys, os
sys.path.append(Path(__file__).parent.parent)
from configs.cpt_config import CPTConfig as CPTConfigClass

# 尝试导入Muon优化器，如果不可用则使用AdamW作为备选
try:
    from muon import Muon
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("警告: Muon优化器不可用，将使用AdamW作为备选")

# 尝试导入Flash Attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("警告: Flash Attention不可用，将使用标准注意力机制")


class CPTDataset(Dataset):
    """继续预训练数据集类"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 对文本进行分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # 对于语言建模，标签就是输入
        }


class CPTTrainer:
    '''
    用来在大量的文本上做继续预训练
    集成了Muon优化器和多种效率优化技术
    '''
    
    def __init__(self, config: CPTConfigClass):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self._setup_logger()
        
        # 初始化模型和分词器
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        self.logger.info(f"正在加载模型: {self.config.model.model_name_or_path}")
        
        # 加载分词器
        self.tokenizerz: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name_or_path,
            trust_remote_code=self.config.model.trust_remote_code,
            cache_dir=self.config.model.cache_dir,
            token=self.config.model.token
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型配置
        config = AutoConfig.from_pretrained(
            self.config.model.model_name_or_path,
            trust_remote_code=self.config.model.trust_remote_code,
            cache_dir=self.config.model.cache_dir,
            token=self.config.model.token
        )
        
        # 确定torch_dtype
        if self.config.model.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.config.model.torch_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.model.trust_remote_code,
            device_map=self.config.model.device_map if torch.cuda.is_available() else None,
            low_cpu_mem_usage=self.config.model.low_cpu_mem_usage,
            token=self.config.model.token
        )
        
        # 启用梯度检查点以节省内存
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.logger.info("模型和分词器加载完成")
    
    def create_optimizer(self):
        """创建优化器"""
        if self.config.training.optim == "muon" and MUON_AVAILABLE:
            self.logger.info("使用Muon优化器")
            self.optimizer = Muon(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            if self.config.training.optim == "muon" and not MUON_AVAILABLE:
                self.logger.warning("Muon优化器不可用，使用AdamW作为备选")
            
            import torch.optim as optim
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
                eps=self.config.training.adam_epsilon
            )
        
        self.logger.info(f"优化器创建完成: {type(self.optimizer).__name__}")
    
    def load_data(self, texts: List[str]) -> CPTDataset:
        """加载和预处理数据"""
        self.logger.info(f"正在加载数据，共 {len(texts)} 条文本")
        
        # 如果设置了最大样本数，则截取数据
        if self.config.data.max_train_samples and len(texts) > self.config.data.max_train_samples:
            texts = texts[:self.config.data.max_train_samples]
            self.logger.info(f"数据已截取到 {len(texts)} 条")
        
        dataset = CPTDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.config.model.max_length
        )
        
        self.logger.info(f"数据集创建完成，共 {len(dataset)} 个样本")
        return dataset
    
    def create_data_loader(self, dataset: CPTDataset, is_training: bool = True) -> DataLoader:
        """创建数据加载器"""
        batch_size = (self.config.training.per_device_train_batch_size if is_training 
                     else self.config.training.per_device_eval_batch_size)
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=self.config.training.dataloader_pin_memory,
            drop_last=self.config.training.dataloader_drop_last if is_training else False
        )
        
        return data_loader
    
    def train_epoch(self, train_loader: DataLoader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for step, batch in enumerate(progress_bar):
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
            
            # 优化器步骤
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (step + 1):.4f}'
            })
            
            # 定期记录日志
            if self.global_step % self.config.training.logging_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}, Loss: {loss.item():.4f}, "
                    f"Avg Loss: {total_loss / (step + 1):.4f}"
                )
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {self.epoch} 完成，平均损失: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(self, eval_loader: DataLoader):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(eval_loader)
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"评估完成，平均损失: {avg_loss:.4f}")
        return avg_loss
    
    def save_model(self, save_path: str):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存配置
        config_path = os.path.join(save_path, "cpt_config.json")
        self.config.save_to_file(config_path)
        
        self.logger.info(f"模型已保存到: {save_path}")
    
    def train(self, texts: List[str], eval_texts: Optional[List[str]] = None):
        """主训练函数"""
        self.logger.info("开始继续预训练")
        
        # 加载模型和分词器
        self.load_model_and_tokenizer()
        
        # 创建优化器
        self.create_optimizer()
        
        # 加载数据
        train_dataset = self.load_data(texts)
        train_loader = self.create_data_loader(train_dataset, is_training=True)
        
        eval_loader = None
        if eval_texts:
            eval_dataset = self.load_data(eval_texts)
            eval_loader = self.create_data_loader(eval_dataset, is_training=False)
        
        # 创建学习率调度器
        total_steps = len(train_loader) * self.config.training.num_train_epochs
        warmup_steps = int(total_steps * self.config.training.warmup_ratio) if self.config.training.warmup_ratio > 0 else self.config.training.warmup_steps
        
        if self.config.training.lr_scheduler_type == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
        elif self.config.training.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps
            )
        elif self.config.training.lr_scheduler_type == "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
        
        # 训练循环
        best_loss = float('inf')
        
        for epoch in range(self.config.training.num_train_epochs):
            self.epoch = epoch
            self.logger.info(f"开始训练 Epoch {epoch + 1}/{self.config.training.num_train_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 评估
            if eval_loader:
                eval_loss = self.evaluate(eval_loader)
                self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
                
                # 保存最佳模型
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.save_model(os.path.join(self.config.training.output_dir, "best_model"))
            else:
                self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % 1 == 0:  # 每个epoch保存一次
                checkpoint_path = os.path.join(self.config.training.output_dir, f"checkpoint-epoch-{epoch + 1}")
                self.save_model(checkpoint_path)
            
            # 更新学习率
            scheduler.step()
        
        # 保存最终模型
        final_model_path = os.path.join(self.config.training.output_dir, "final_model")
        self.save_model(final_model_path)
        
        self.logger.info("训练完成！")
        return best_loss if eval_loader else train_loss


# 使用示例
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="继续预训练脚本")
    parser.add_argument("--config", type=str, default="configs/cpt_config.json", help="配置文件路径")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径")
    parser.add_argument("--eval_data", type=str, help="评估数据路径")
    parser.add_argument("--model_name_or_path", type=str, help="模型路径，覆盖配置文件中的设置")
    parser.add_argument("--output_dir", type=str, help="输出目录，覆盖配置文件中的设置")
    parser.add_argument("--learning_rate", type=float, help="学习率，覆盖配置文件中的设置")
    parser.add_argument("--batch_size", type=int, help="批大小，覆盖配置文件中的设置")
    parser.add_argument("--num_train_epochs", type=int, help="训练轮数，覆盖配置文件中的设置")
    parser.add_argument("--max_length", type=int, help="最大序列长度，覆盖配置文件中的设置")
    
    args = parser.parse_args()
    
    # 加载配置
    config = CPTConfigClass.from_file(args.config)
    
    # 覆盖配置
    if args.model_name_or_path:
        config.model.model_name_or_path = args.model_name_or_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.num_train_epochs:
        config.training.num_train_epochs = args.num_train_epochs
    if args.max_length:
        config.model.max_length = args.max_length
    
    # 加载数据
    with open(args.train_data, 'r', encoding='utf-8') as f:
        train_texts = [line.strip() for line in f if line.strip()]
    
    eval_texts = None
    if args.eval_data:
        with open(args.eval_data, 'r', encoding='utf-8') as f:
            eval_texts = [line.strip() for line in f if line.strip()]
    
    # 创建训练器并开始训练
    trainer = CPTTrainer(config)
    trainer.train(train_texts, eval_texts)


if __name__ == "__main__":
    main()