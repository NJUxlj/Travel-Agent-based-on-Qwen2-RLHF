# 自定义数据集类
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class TravelDialogueDataset(Dataset):
    """
    旅游对话数据集类
    """
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """
        初始化数据集
        
        Args:
            data: 对话数据列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx: 数据索引
        
        Returns:
            包含输入输出的字典
        """
        item = self.data[idx]
        
        # 构建提示模板
        prompt = f"你是一个专业的旅游顾问。请根据用户的问题提供专业的建议。\n\n用户：{item['question']}\n\n助手："
        target = item['answer']
        
        # 构建完整文本
        full_text = prompt + target
        
        # 编码
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 准备输入标签
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # 创建标签（将prompt部分的标签设为-100）
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        prompt_length = (prompt_encoding["input_ids"] != self.tokenizer.pad_token_id).sum()
        
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class TravelPreferenceDataset(Dataset):
    """
    用于DPO训练的偏好数据集
    """
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """
        初始化数据集
        
        Args:
            data: 偏好数据列表，每项包含question, chosen和rejected
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个偏好数据样本
        
        Args:
            idx: 数据索引
        
        Returns:
            包含prompt和偏好选择的字典
        """
        item = self.data[idx]
        
        # 构建提示模板
        prompt = f"你是一个专业的旅游顾问。请根据用户的问题提供专业的建议。\n\n用户：{item['question']}\n\n助手："
        
        # 编码prompt
        prompt_encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码chosen和rejected回答
        chosen_encodings = self.tokenizer(
            item['chosen'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_encodings = self.tokenizer(
            item['rejected'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "prompt_ids": prompt_encodings["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_encodings["attention_mask"].squeeze(0),
            "chosen_ids": chosen_encodings["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encodings["attention_mask"].squeeze(0),
            "rejected_ids": rejected_encodings["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encodings["attention_mask"].squeeze(0)
        }