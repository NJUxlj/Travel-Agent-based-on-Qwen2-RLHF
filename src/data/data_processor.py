# 数据处理脚本
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from datasets import (
    load_dataset,
    load_from_disk,
)


import sys
sys.path.append("../../")  # 添加上级目录的上级目录到sys.path
sys.path.append("../")
from utils import (
    get_max_length_from_model,
)

class DataProcessor:
    """
    处理旅行对话数据的类，支持普通对话数据和DPO偏好数据的处理
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        system_prompt: str = "You are a helpful AI travel agent. Help users plan their trips and provide travel advice.",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def _format_conversation(
        self,
        messages: List[Dict[str, str]],
        include_system_prompt: bool = True
    ) -> str:
        """
        将对话消息列表格式化为单个字符串
        
        Args:
            messages: 消息列表，每个消息包含'role'和'content'
            include_system_prompt: 是否包含系统提示
            
        Returns:
            格式化后的对话字符串
        """
        conversation = []
        if include_system_prompt:
            conversation.append(f"<|system|>{self.system_prompt}")
            
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'user':
                conversation.append(f"<|user|>{content}")
            elif role == 'assistant':
                conversation.append(f"<|assistant|>{content}")
                
        return "\n".join(conversation)
    
    def _tokenize_function(
        self,
        examples: Dict[str, List],
        text_column: str = "text",
        add_eos_token: bool = True
    ) -> Dict[str, List]:
        """
        对文本进行分词处理
        
        Args:
            examples: 包含文本的字典
            text_column: 文本列的名称
            add_eos_token: 是否添加EOS标记
            
        Returns:
            包含tokenized结果的字典
        """
        texts = examples[text_column]
        
        # 批量tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        if add_eos_token:
            for i in range(len(tokenized['input_ids'])):
                # padding
                if len(tokenized['input_ids'][i]) < self.max_length:
                    tokenized['input_ids'][i].append(self.tokenizer.eos_token_id)
                    tokenized['attention_mask'][i].append(1)
                    
        return tokenized
    
    def process_conversation_data(
        self,
        data_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        train_ratio: float = 0.9,
        use_huggingface_format: bool = True,
    ) -> DatasetDict:
        """
        处理对话数据用于LoRA微调
        
        Args:
            data_path: 原始数据路径
            output_path: 处理后数据的保存路径
            train_ratio: 训练集比例
            
        Returns:
            处理后的数据集
        """
        self.logger.info(f"Processing conversation data from {data_path}")
        
        
        formatted_conversations = []

        # 读取原始数据(json格式)
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # 格式化对话
        # raw_data: List[Dict]
        for conversation in tqdm(raw_data, desc="Formatting conversations"):
            formatted_text = self._format_conversation(conversation['messages'])
            formatted_conversations.append({
                'text': formatted_text,
                'id': conversation.get('id', len(formatted_conversations))
            })
            
        # 创建数据集
        dataset = Dataset.from_pandas(pd.DataFrame(formatted_conversations))
        
        # 分词处理
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing texts"
        )
        
        # 划分训练集和验证集
        split_dataset = tokenized_dataset.train_test_split(
            train_size=train_ratio,
            shuffle=True,
            seed=42
        )
        
        # 保存处理后的数据
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            split_dataset.save_to_disk(output_path)
            self.logger.info(f"Saved processed dataset to {output_path}")
            
        return split_dataset
    
    def process_dpo_data(
        self,
        data_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        train_ratio: float = 0.9,
    ) -> DatasetDict:
        """
        处理偏好数据用于DPO训练
        
        Args:
            data_path: 原始数据路径
            output_path: 处理后数据的保存路径
            train_ratio: 训练集比例
            
        Returns:
            处理后的数据集
        """
        self.logger.info(f"Processing DPO data from {data_path}")
        
        # 读取原始数据
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # 处理DPO数据
        processed_data = []
        for item in tqdm(raw_data, desc="Processing DPO data"):
            # 格式化提示词
            prompt = self._format_conversation(
                item['prompt_messages'],
                include_system_prompt=True
            )
            
            # 处理选中和被拒绝的回复
            chosen = item['chosen_message']['content']
            rejected = item['rejected_message']['content']
            
            processed_data.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'id': item.get('id', len(processed_data))
            })
            
        # 创建数据集
        dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
        
        # 划分训练集和验证集
        split_dataset = dataset.train_test_split(
            train_size=train_ratio,
            shuffle=True,
            seed=42
        )
        
        # 保存处理后的数据
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            split_dataset.save_to_disk(output_path)
            self.logger.info(f"Saved processed DPO dataset to {output_path}")
            
        return split_dataset
    
    def process_crosswoz_data(self, file_path):  
        """处理CrossWOZ数据集"""  
        formatted_data = []  
        
        with open(file_path, 'r', encoding='utf-8') as f:  
            raw_data = json.load(f)  
        
        for dialogue_id, dialogue in raw_data.items():  
            messages = []  
            for turn in dialogue['messages']:  
                if turn['role'] in ['user', 'system']:  
                    messages.append({  
                        "role": "user" if turn['role'] == 'user' else "assistant",  
                        "content": turn['content']  
                    })  
            
            if len(messages) >= 2:  # 确保至少有一轮对话  
                formatted_data.append({  
                    "messages": messages,  
                    "id": f"crosswoz_{dialogue_id}"  
                })  

        # 保存处理后的数据  
        with open('processed_crosswoz.json', 'w', encoding='utf-8') as f:  
            json.dump(formatted_data, f, ensure_ascii=False, indent=2) 
        
        print("已保存处理后的数据集CrossWOZ 到 processed_crosswoz.json ~~~")
        
        return formatted_data 
    @staticmethod
    def validate_data_format(data_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        验证数据格式是否正确
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                return False, "数据必须是列表格式"
                
            for item in data:
                if 'messages' not in item:
                    return False, "每个对话必须包含'messages'字段"
                    
                for message in item['messages']:
                    if 'role' not in message or 'content' not in message:
                        return False, "每条消息必须包含'role'和'content'字段"
                        
            return True, "数据格式正确"
            
        except json.JSONDecodeError:
            return False, "无效的JSON格式"
        except Exception as e:
            return False, f"验证过程出错: {str(e)}"
    
    def prepare_example_data(self) -> Dict[str, List[Dict]]:
        """
        生成示例数据格式
        
        Returns:
            包含示例数据的字典
        """
        # 普通对话数据示例
        conversation_example = [
            {
                "messages": [
                    {"role": "user", "content": "我想去北京旅游，有什么建议吗？"},
                    {"role": "assistant", "content": "北京是一个历史文化名城，有很多著名景点..."},
                    {"role": "user", "content": "故宫要怎么玩？"},
                    {"role": "assistant", "content": "参观故宫建议从午门进入，按照中轴线参观..."}
                ],
                "id": "conv_001"
            }
        ]
        
        # DPO数据示例
        dpo_example = [
            {
                "prompt_messages": [
                    {"role": "user", "content": "推荐一个适合冬天旅游的地方"}
                ],
                "chosen_message": {
                    "role": "assistant",
                    "content": "我建议您考虑去海南三亚。冬季气候宜人，可以享受阳光沙滩..."
                },
                "rejected_message": {
                    "role": "assistant",
                    "content": "三亚就还不错吧，那里冬天也挺暖和的。"
                },
                "id": "dpo_001"
            }
        ]
        
        return {
            "conversation_data": conversation_example,
            "dpo_data": dpo_example
        }
        
        
        




if __name__ == '__main__':
    pass