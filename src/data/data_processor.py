# 数据处理脚本
import json
import os
import copy
import logging
import pandas as pd
from datetime import datetime  
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datasets import Dataset, DatasetDict
from huggingface_hub import (
    HfApi, 
    DatasetCard, 
    DatasetCardData,
    create_repo,   
    upload_file,
)  


from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
)
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
        logging.basicConfig(  
            level=logging.INFO,  
            format='%(asctime)s - %(levelname)s - %(message)s'  
        )  
        # self.logger.setLevel(logging.INFO)
    
    def load_from_huggingface(
        self,
        dataset_name: str,
        split: str = None,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> Dataset:
        """
        从Hugging Face Hub加载数据集

        Args:
            dataset_name: 数据集名称
            split: 数据集分割
            use_auth_token: 使用的认证令牌
        """
        self.logger.info(f"Loading dataset {dataset_name} from Hugging Face Hub")
        dataset = load_dataset(
            dataset_name,
            split=split,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
        )
        self.logger.info(f"Loaded dataset {dataset_name} with {len(dataset)} samples, split = {split}")
        return dataset

        
    def _format_conversation(
        self,
        examples: Dict[str, List],
        messages: List[Dict[str, str]] = None,
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
        
        results = {
            "input": [],
            "labels": [],
        }
        
        # print("************************************")
        # print("examples = \n", examples)
        # print("*************************************")        

        if messages != None:
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
        
        else:
            
            for index in range(len(examples['content'])):
                content = examples['content'][index]
                role = examples['role'][index]
                if role == 'usr':
                    results['input'].append(f'{role}: {content}')
                elif role == 'sys':
                    results['labels'].append(f'{role}: {content}')
            
            return results
    
    def _tokenize_function(
        self,
        examples: Dict[str, List],
        text_column: str = "input",
        label_column: str = "labels",
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
        labels = examples[label_column]
        
        result = {  
            "input_ids": [],  
            "attention_mask": [],  
            "labels": []  
        }  
        
        
        for text, label in zip(texts, labels):  
            # 构建完整的对话文本  
            full_text = f"{text}\n\n{label}"  
            
            # 对完整文本进行编码  
            encoded = self.tokenizer(  
                full_text,  
                truncation=True,  
                max_length=self.max_length,  
                padding="max_length",  
                return_tensors=None,  
            )  
            
            # 获取input_ids和attention_mask  
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            
            # 对于因果语言建模，labels应该与input_ids相同，但向右移动一位  
            # 这样每个位置都预测下一个token  
            labels = copy.deepcopy(input_ids) 
            labels[:-1] = input_ids[1:]  # 向左移动一位  
            labels[-1] = -100  # 最后一个token没有下一个token可预测  
            
            # 将padding tokens的label设为-100  
            labels[attention_mask == 0] = -100  
            
            '''
            在PyTorch和Transformers库中，-100是一个特殊的值，用作损失计算时的忽略索引（ignore_index）。设置为-100的原因是：

            技术原因：

            在计算交叉熵损失（CrossEntropyLoss）时，-100是默认的ignore_index值
            当标签为-100时，这些位置的预测不会参与损失计算
            '''
            
            # 只将padding tokens的label设为-100  
            labels[attention_mask == 0] = -100   
            
            result["input_ids"].append(input_ids)  
            result["attention_mask"].append(attention_mask)  
            result["labels"].append(labels)  
        
        return result
            
        
        # 批量tokenize
        # tokenized_input = self.tokenizer(
        #     texts,
        #     truncation=True,
        #     max_length=self.max_length,
        #     padding="max_length",
        #     return_tensors=None, # data_collator会负责转Tensor的
        # )
        
        # tokenized_labels = self.tokenizer(
        #     labels,
        #     truncation=True,
        #     max_length=self.max_length,
        #     padding="max_length",
        #     return_tensors=None,
        # )
    
        # 将标签中的padding token替换为-100  
        # label_ids = tokenized_labels["input_ids"]  
        # label_attention_mask = tokenized_labels["attention_mask"]  
        # for i in range(len(label_ids)):  
        #     if label_attention_mask[i] == 0:  
        #         label_ids[i] = -100  


        # tokenized_input['labels'] = tokenized_labels['input_ids']
        # tokenized_input['labels_attention_mask'] = tokenized_labels['attention_mask']
                    
        # return tokenized_input
    
    def process_conversation_data_json(
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
    
    
    def process_conversation_data_huggingface(
        self,
        data_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        train_ratio: float = 0.9,
        use_huggingface_format: bool = True,
    ) -> DatasetDict:
        """
        处理对话数据用于LoRA微调

        Args:
        """
        
        self.logger.info(f"Processing conversation data from {data_path}")
        # 加载数据集
        dataset = self.load_from_huggingface(data_path)
        
        print("dataset.keys = ", dataset.keys())
        
        for split in ['train', 'validation', 'test']:
            # 格式化对话
            dataset[split] = dataset[split].map(
                lambda x: self._format_conversation(x),
                batched=True,
                remove_columns=dataset[split].column_names,
                desc=f"Formatting conversations for {split} of dataset {data_path}"
            )

            # 分词处理
            dataset[split] = dataset[split].map(
                self._tokenize_function,
                batched=True,
                remove_columns=dataset[split].column_names,
                desc=f"Tokenizing texts for {split} of dataset {data_path}"
            )
        
        
        
        return dataset
        
        
    
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
                if turn['role'] in ['user', 'sys']:  
                    messages.append({  
                        "role": "user" if turn['role'] == 'usr' else "sys",  
                        "content": turn['content']  
                    })  
            
            if len(messages) >= 2:  # 确保至少有一轮对话  
                formatted_data.append({  
                    "messages": messages,  
                    "id": f"crosswoz_{dialogue_id}"  
                })  

        # 保存处理后的数据  
        store_path = '/root/autodl-tmp/Travel-Agent-based-on-LLM-and-SFT/src/data/processed_data/processed_crosswoz.json'
        print("处理后的数据集的保存路径是：",store_path)
        with open(store_path, 'w', encoding='utf-8') as f:  
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
        
    def load_dataset_from_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_pandas(pd.DataFrame(data))
    
    
    
    def print_dataset_example(dataset):
        print("\n\n**************** print dataset example********************* ")
        print(dataset[0])
        print("****************************************************\n\n")
        
        
    def upload_to_huggingface(self, dataset: Dataset, repo_id: str):
        """
        将数据集上传到Hugging Face Hub

        Args:
            dataset: 数据集
            repo_id: 仓库ID
        """
        dataset.push_to_hub(repo_id)
        self.logger.info(f"Dataset uploaded to {repo_id}")
        
        



class CrossWOZProcessor(DataProcessor):  
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        system_prompt: str = "You are a helpful AI travel agent. Help users plan their trips and provide travel advice.",
        ):  
        super().__init__(tokenizer, max_length, system_prompt)
        
    def _flatten_dialog(self, dialog_id: str, dialog_data: Dict) -> List[Dict]:  
        """  
        将单个对话展平为多个训练样本  
        
        Args:  
            dialog_id: 对话ID  
            dialog_data: 单个对话的原始数据  
            
        Returns:  
            List[Dict]: 展平后的训练样本列表  
        """  
        flattened_data = []  
        messages = dialog_data.get('messages', [])  
        
        # 构建对话历史  
        history = []  
        for i, msg in enumerate(messages):  
            current_sample = {  
                'dialog_id': dialog_id,  
                'turn_id': i,  
                'role': msg['role'],  
                'content': msg['content'],  
                'dialog_act': json.dumps(msg.get('dialog_act', []), ensure_ascii=False),  
                'history': json.dumps(history.copy(), ensure_ascii=False),  
            }  
            
            # 添加用户状态和系统状态  
            if msg['role'] == 'usr':  
                current_sample['user_state'] = json.dumps(msg.get('user_state', []), ensure_ascii=False)  
            else:  # sys  
                current_sample['sys_state'] = json.dumps(msg.get('sys_state', {}), ensure_ascii=False)  
                current_sample['sys_state_init'] = json.dumps(msg.get('sys_state_init', {}), ensure_ascii=False)  
            
            # 更新历史  
            history.append({  
                'role': msg['role'],  
                'content': msg['content']  
            })  
            
            # 添加目标信息  
            if i == 0:  # 只在第一轮添加目标信息  
                current_sample['goal'] = json.dumps(dialog_data.get('goal', []), ensure_ascii=False)  
                current_sample['sys_usr'] = json.dumps(dialog_data.get('sys-usr', []), ensure_ascii=False)  
            
            flattened_data.append(current_sample)  
            
        return flattened_data  

    def load_dataset_from_json(self, json_path: str) -> Dataset:  
        """  
        从JSON文件加载CrossWOZ数据集  
        
        Args:  
            json_path: JSON文件路径  
            
        Returns:  
            Dataset: 处理后的Hugging Face数据集  
        """  
        try:  
            # 读取JSON文件  
            self.logger.info(f"Loading data from {json_path}")  
            with open(json_path, 'r', encoding='utf-8') as f:  
                raw_data = json.load(f)  
            
            # 展平所有对话  
            flattened_data = []  
            for dialog_id, dialog_content in tqdm(raw_data.items(), desc="Processing dialogs"):  
                flattened_data.extend(self._flatten_dialog(dialog_id, dialog_content))  
            
            # 转换为DataFrame  
            df = pd.DataFrame(flattened_data)  
            
            # 填充缺失值  
            df = df.fillna("")  
            
            # 记录数据集信息  
            self.logger.info(f"Created dataset with {len(df)} samples")  
            self.logger.info(f"Columns: {df.columns.tolist()}")  
            
            # 转换为Dataset格式  
            dataset = Dataset.from_pandas(df)  
            
            return dataset  
            
        except Exception as e:  
            self.logger.error(f"Error loading dataset: {str(e)}")  
            raise  

    def prepare_training_features(self, dataset: Dataset, max_length: int = 512) -> Dataset:  
        """  
        准备用于训练的特征  
        
        Args:  
            dataset: 原始数据集  
            tokenizer: 分词器  
            max_length: 最大序列长度  
            
        Returns:  
            Dataset: 处理后的数据集  
        """  
        def _prepare_single_sample(example):  
            # 构建输入文本  
            if example['role'] == 'usr':  
                input_text = f"用户：{example['content']}"  
                # 如果有上文，添加到输入中  
                history = json.loads(example['history'])  
                if history:  
                    context = "\n".join([  
                        f"{'用户' if msg['role']=='usr' else '助手'}：{msg['content']}"  
                        for msg in history  
                    ])  
                    input_text = f"{context}\n{input_text}"  
            else:  
                input_text = f"助手：{example['content']}"  
            
            # tokenize  
            tokenized = self.tokenizer(  
                input_text,  
                truncation=True,  
                max_length=max_length,  
                padding='max_length',  
                return_tensors='pt'  
            )  
            
            return {  
                'input_ids': tokenized['input_ids'][0],  
                'attention_mask': tokenized['attention_mask'][0],  
                'labels': tokenized['input_ids'][0] if example['role'] == 'sys' else None  
            }  
        
        return dataset.map(  
            _prepare_single_sample,  
            remove_columns=dataset.column_names,  
            load_from_cache_file=False  
        )  






class DatasetPublisher:  
    def __init__(  
        self,  
        dataset_name: str,  
        organization: Optional[str] = None,  
        token: Optional[str] = None  
    ):  
        """  
        初始化数据集发布器  
        
        Args:  
            dataset_name: 数据集名称  
            organization: 组织名称（可选）  
            token: Hugging Face token（可选，建议设置环境变量 HF_TOKEN）  
        """  
        self.logger = logging.getLogger(__name__)  
        self.dataset_name = dataset_name  
        self.organization = organization  
        self.token = token or os.getenv("HF_TOKEN")  
        self.api = HfApi(token=self.token)  
        
        # 设置完整的仓库ID  
        self.repo_id = f"{organization}/{dataset_name}" if organization else dataset_name  
        
    def _create_dataset_card(  
        self,  
        dataset: Dataset,  
        description: str,  
        language: str = "zh",  
        license: str = "MIT",  
        tags: Optional[list] = None  
    ) -> DatasetCard:  
        """  
        创建数据集卡片  
        """  
        split_sizes = {split: len(dataset) for split, dataset in dataset.items()} 
        
        dataset_name =  os.path.basename(self.dataset_name)
        # 基础信息  
        card_data = DatasetCardData(  
            language=language,  
            license=license,  
            pretty_name=dataset_name,  
            # tags=tags or ["conversational", "dialogue", "chinese", "travel", "agent", "crosswoz"],  
            task_categories=["language-modeling"], 
            task_ids= ["conversation", "question-answering", "dialogue-modeling"],
            size_categories=["1K<n<10K"],  
            multilinguality="monolingual",
            # languages=["zh"],  
        )  
        
        # 创建卡片  
        card = DatasetCard.from_template(
                card_data,
                pretty_name = card_data.pretty_name
            )  
        
        # 添加详细描述  
        card.content = f"""---  
                    annotations_creators:  
                    - expert-generated  
                    language:  
                    - zh  
                    language_creators:  
                    - expert-generated  
                    license:  
                    - {license}  
                    multilinguality:  
                    - monolingual  
                    pretty_name: {self.dataset_name}  
                    size_categories:  
                    - 1K<n<10K  
                    source_datasets:  
                    - original  
                    task_categories:  
                    - conversational  
                    task_ids:  
                    - dialogue-modeling  
                    tags:  
                    - crosswoz  
                    - dialogue  
                    - chinese  

                    dataset_info:  
                    name: {self.dataset_name}  
                    description: |  
                        {description}  
  
                        statistics:  
                            splits:  
                        {json.dumps(split_sizes, indent=2, ensure_ascii=False).replace('{', '').replace('}', '').replace('"', '')}  
                            features:  
                            - name: dialog_id  
                                dtype: string  
                            - name: turn_id  
                                dtype: int64  
                            - name: role  
                                dtype: string  
                            - name: content  
                                dtype: string  
                            - name: dialog_act  
                                dtype: string  
                            - name: history  
                                dtype: string  
                            - name: user_state  
                                dtype: string  
                            - name: goal  
                                dtype: string  
                            - name: sys_usr  
                                dtype: string  
                            - name: sys_state  
                                dtype: string  
                            - name: sys_state_init  
                                dtype: string  


                example:  
                    train:
                {json.dumps(dataset['train'][0], indent=6, ensure_ascii=False)}  

                preprocessing:  
                    steps:  
                    - 对话历史的展平和格式化  
                    - 状态信息的JSON序列化  
                    - 特殊字段的标准化处理  
                    - 数据集的划分（训练集/验证集/测试集）  

                usage:  
                loading_code: |  
                    ```python  
                    from datasets import load_dataset  

                    # 加载完整数据集  
                    dataset = load_dataset("{self.repo_id}")  

                    # 加载特定分割  
                    train_dataset = load_dataset("{self.repo_id}", split="train")  
                    validation_dataset = load_dataset("{self.repo_id}", split="validation")  
                    test_dataset = load_dataset("{self.repo_id}", split="test")  
                    ```  

        splits:  
            train: {split_sizes.get('train', 'N/A')} samples  
            validation: {split_sizes.get('validation', 'N/A')} samples  
            test: {split_sizes.get('test', 'N/A')} samples  

        citation: |  
        @inproceedings{{zhu2020crosswoz,  
            title={{CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset}},  
            author={{Zhu, Qi and Zhang, Zheng and Fang, Yan and Li, Xiang and Takanobu, Ryuichi and Li, Jinchao and Peng, Baolin and Gao, Jianfeng and Zhu, Xiaoyan and Huang, Minlie}},  
            booktitle={{Transactions of the Association for Computational Linguistics}},  
            year={{2020}},  
            url={{https://arxiv.org/abs/2002.11893}}  
        }}  

        version:  
        version_name: "1.0.0"  
        changes:  
            - {datetime.now().strftime('%Y-%m-%d')}: 首次发布  
        
        
        """
        return card.content
    
    def push_to_hub(
        self,
        dataset: Dataset,
        description: str,
        private: bool = False,
        token: Optional[str] = None
        ) -> str:
        """
        将数据集推送到 Hugging Face Hub

        ini
        Args:  
            dataset: 要推送的数据集  
            description: 数据集描述  
            private: 是否为私有仓库  
            token: Hugging Face token（可选）  
            
        Returns:  
            str: 数据集在Hub上的URL  
        """  
        
        try:  
            # 创建数据集卡片  
            card_content = self._create_dataset_card(dataset, description)  
            
            # 创建DatasetDict  
            # dataset_dict = DatasetDict({  
            #     "train": dataset  
            # })  
            
            # 推送到Hub  
            self.logger.info(f"Pushing dataset to {self.repo_id}")  
            
            # 创建仓库（如果不存在）  
            from huggingface_hub import create_repo  
            try:  
                create_repo(  
                    repo_id=self.repo_id,  
                    repo_type="dataset",  
                    private=private,  
                    token=token or self.token,  
                    exist_ok=True  
                )  
            except Exception as e:  
                self.logger.warning(f"Repository creation warning (可能已存在): {str(e)}")  
                
                
            dataset.push_to_hub(  
                self.repo_id,  
                token=token or self.token,  
                private=private  
            )  
            
            # 单独更新README  
            from huggingface_hub import upload_file  
            dirname = os.path.dirname(__file__)
            readme_path = os.path.join(dirname, "README.md")  
            with open(readme_path, "w", encoding="utf-8") as f:  
                f.write(card_content)  
            
            upload_file(  
                path_or_fileobj=readme_path,  
                path_in_repo="README.md",  
                repo_id=self.repo_id,  
                repo_type="dataset",  
                token=token or self.token  
            )  
            
            url = f"https://huggingface.co/datasets/{self.repo_id}"  
            self.logger.info(f"Dataset successfully pushed to {url}")  
            return url  
            
        except Exception as e:  
            self.logger.error(f"Error pushing dataset to hub: {str(e)}")  
            raise  
        
    
def push_crosswoz_to_hub(model_name, dataset_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = get_max_length_from_model(model_name)
    processor = CrossWOZProcessor(tokenizer, max_length)
    
    # dataset = processor.load_dataset_from_json(dataset_name)
    parent = os.path.dirname(dataset_name)
    train_dir = os.path.join(parent, "train.json")
    valid_dir = os.path.join(parent, "val.json")
    test_dir = os.path.join(parent, "test.json")
    # 加载所有数据集分割  
    datasets = {  
        'train': processor.load_dataset_from_json(train_dir),  
        'validation': processor.load_dataset_from_json(valid_dir),  
        'test': processor.load_dataset_from_json(test_dir)  
    }  

    # 创建DatasetDict  
    dataset_dict = DatasetDict(datasets)  
    
    token = os.environ.get("HF_TOKEN")  
    if not token:  
        raise ValueError("Please set HF_TOKEN environment variable")  
    
    # 初始化发布器  
    publisher = DatasetPublisher(  
        dataset_name="crosswoz-sft",  
        organization="BruceNju",  # 替换为你的组织名  
        token=token  # 替换为你的token，或设置环境变量HF_TOKEN (None是默认) 
    )  

    # 数据集描述  
    description = """  
    这是一个基于CrossWOZ数据集处理的对话数据集，专门用于大模型的监督微调（SFT）任务。  
    数据集包含多轮对话、用户目标、对话状态等信息，适合训练任务型对话系统。  

    原始数据来源于CrossWOZ项目，经过专门的预处理使其更适合现代大模型训练。  
    
    ## 核心特征：
    ---
    这是首个大规模的中文跨域任务型对话数据集
    包含6,012个对话，102,000个话语，覆盖5个领域(酒店、餐厅、景点、地铁和出租车)
    约60%的对话包含跨域用户目标
    主要创新点：

    更具挑战性的域间依赖关系：
    - 一个领域的选择会动态影响其他相关领域的选择
    - 例如用户选择的景点会影响后续酒店的推荐范围(需要在景点附近)
    
    完整的标注：
    - 同时提供用户端和系统端的对话状态标注
    - 包含对话行为(dialogue acts)的标注
    - 用户状态标注有助于追踪对话流程和建模用户行为
    
    高质量的数据收集：
    - 采用同步对话收集方式
    - 两个经过训练的工作人员实时对话
    - 相比MultiWOZ的异步方式，可以确保对话的连贯性
    
    数据规模：
    ---
    - 训练集：5,012个对话
    - 验证集：500个对话
    - 测试集：500个对话
    平均每个对话包含3.24个子目标，16.9个回合
    对话复杂度高于MultiWOZ等现有数据集
    
    这个数据集为研究跨域对话系统提供了新的测试平台，可用于对话状态追踪、策略学习等多个任务的研究。
    """  

    # 推送到Hub  
    url = publisher.push_to_hub(  
        dataset=dataset_dict,  
        description=description,  
        private=False  # 设置为True如果想创建私有数据集  
    )  

    print(f"Dataset published successfully at: {url}") 
    
    

if __name__ == '__main__':
    model_name = "/root/autodl-tmp/models/Qwen2.5-1.5B"
    dataset_name = "/root/autodl-tmp/Travel-Agent-based-on-LLM-and-SFT/data/raw/crosswoz/train.json"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # max_length = get_max_length_from_model(model_name)
    # processor = CrossWOZProcessor(tokenizer, max_length)
    
    # dataset = processor.load_dataset_from_json(dataset_name)
    
    
    # push_crosswoz_to_hub(model_name, dataset_name)
    