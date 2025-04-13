import torch  
import torch.nn as nn  
import os  
from torch.utils.data import Dataset  
from transformers import (  
    AutoTokenizer,  
    Qwen2ForCausalLM,  
    TrainingArguments,  
    Trainer,  
    BitsAndBytesConfig  
)  
from peft import LoraConfig, get_peft_model  
from datasets import load_dataset, DatasetDict
from typing import Optional, Dict, List, Union, Tuple  
import torch.nn.functional as F  
from dataclasses import dataclass  
import numpy as np  
import copy  
from torch.nn.utils.rnn import pad_sequence  

from src.configs.config import (  
    MODEL_PATH,  
    DATA_PATH,
    DPO_DATA_PATH,  
    CACHED_DPO_DATA_PATH,  
    GRPO_MODEL_PATH,
    CACHED_GRPO_DATA_PATH,
)  


from src.evaluation.qa_evaluate import QAEvaluator


'''

GRPO算法的核心优势在于：

    无需critic模型，通过组采样方式估计advantage
    在每个prompt上生成多个输出，使用组内响应的相对得分计算advantage
    使用KL惩罚项保持与参考模型的接近度
    
    
GRPO核心思想：
    与PPO不同，GRPO不使用critic网络来估计值函数。
    相反，它使用同一prompt生成的多个响应（一个组）来计算相对优势。
    组内的每个响应与该组的平均得分进行比较，得到相对优势。

优势归一化：
    优势通过组内的均值和标准差进行归一化（可选），这有助于稳定训练，防止高方差。
    公式：(r_i - mean(r)) / std(r)

KL散度惩罚：
    为防止策略偏离太远，添加了KL散度惩罚项，这与PPO中的信任区域约束相似。

数据处理：
    代码改进了数据处理方式，将prompt和response分开，这有助于更精确地计算响应生成的概率。

'''


class QADataset(torch.utils.data.Dataset):  
    """  
    用于问答评估的数据集类  
    """  
    def __init__(self, dataset):  
        self.dataset = dataset  
        self.input_ids = dataset["input_ids"]  
        self.attention_mask = dataset["attention_mask"]  
        self.labels = dataset["labels"]  
    
    def __len__(self):  
        return len(self.input_ids)  
    
    def __getitem__(self, idx):  
        item = {  
            "input_ids": self.input_ids[idx],  
            "attention_mask": self.attention_mask[idx],  
            "labels": self.labels[idx]  
        }  
        return item  




class CustomGRPODataset(Dataset):  
    def __init__(self, tokenized_data):  
        self.data = tokenized_data  
        
    def __len__(self):  
        return len(self.data["input_ids"])  
    
    def __getitem__(self, idx):  
        return {  
            "input_ids": self.data["input_ids"][idx],  
            "attention_mask": self.data["attention_mask"][idx],  
            "response_ids": self.data["response_ids"][idx] if "response_ids" in self.data else None,  
            "group_labels": self.data["group_labels"][idx] if "group_labels" in self.data else None  
        }  


@dataclass  
class GRPOConfig:  
    """Configuration for GRPO training."""  
    beta: float = 0.1  # KL divergence weight  
    group_size: int = 4  # Number of responses per prompt  
    mu: float = 1.0  # Clipping coefficient (1.0 means no clipping)  
    kl_coef: float = 0.1  # KL coefficient  
    scale_rewards: bool = True  # Whether to normalize advantages by std  


class GRPOTrainer(Trainer):  
    def __init__(  
        self,  
        ref_model,
        tokenizer,
        max_seq_length,
        **kwargs  
    ):  
        super().__init__(**kwargs)
        self.ref_model = ref_model
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = self.ref_model.device
        
    def compute_metrics(self, eval_preds):  
        """计算评估指标  
        
        Args:  
            eval_preds: 评估预测结果，包含predictions和label_ids  
            
        Returns:  
            包含多个评估指标的字典  
        """  
        
        evaluator = QAEvaluator(self.model, self.tokenizer, self.max_seq_length)
        
        evaluator.compute_metrics(eval_preds)


    def compute_loss(self, model, inputs, num_items_in_batch=None):
        """计算GRPO损失函数  
        
        GRPO的核心是计算组内样本的相对优势，然后使用这些优势计算损失  
        """  
        input_ids = inputs["input_ids"]  
        attention_mask = inputs["attention_mask"]  
        batch_size = input_ids.shape[0]  
        
        # 计算每组的大小（如果不能整除，最后一组可能会小于group_size）  
        num_groups = (batch_size + self.group_size - 1) // self.group_size  
        
        # 前向传播获取当前策略的logits  
        outputs = model(  
            input_ids=input_ids,  
            attention_mask=attention_mask  
        )  
        logits = outputs.logits  
        
        # 获取labels（输入的下一个token）  
        labels = input_ids.clone()  
        
        # 创建因果mask以确保只考虑自回归条件概率  
        # 找出所有非pad的位置  
        non_padding_mask = labels != self.tokenizer.pad_token_id  
        
        # 计算当前策略的对数概率  
        log_probs = self._get_batch_logprobs(logits, labels, non_padding_mask)  
        
        # 计算参考策略的对数概率  
        with torch.no_grad():  
            ref_outputs = self.ref_model(  
                input_ids=input_ids,  
                attention_mask=attention_mask  
            )  
            ref_logits = ref_outputs.logits  
            ref_log_probs = self._get_batch_logprobs(ref_logits, labels, non_padding_mask)  
        
        # 收集每个组的log_probs和ref_log_probs  
        group_log_probs = []  
        group_ref_log_probs = []  
        
        for i in range(num_groups):  
            start_idx = i * self.group_size  
            end_idx = min((i + 1) * self.group_size, batch_size)  
            
            # 收集当前组的对数概率  
            group_log_probs.append(log_probs[start_idx:end_idx])  
            group_ref_log_probs.append(ref_log_probs[start_idx:end_idx])  
        
        # 计算组内的优势函数  
        advantages = []  
        
        for g_log_probs, g_ref_log_probs in zip(group_log_probs, group_ref_log_probs):  
            # 计算当前组内的KL散度（log_p - log_ref_p）  
            kl_divs = g_log_probs - g_ref_log_probs  
            
            # 计算相对优势：使用每个样本在组内的相对得分  
            group_mean = kl_divs.mean()  
            
            # 计算每个样本的优势（相对于组均值）  
            if self.scale_rewards and len(kl_divs) > 1:  
                group_std = kl_divs.std() + 1e-8  # 添加小的epsilon以防止除零  
                sample_advantages = (kl_divs - group_mean) / group_std  
            else:  
                sample_advantages = kl_divs - group_mean  
                
            advantages.append(sample_advantages)  
        
        # 将所有优势合并回一个张量  
        all_advantages = torch.cat(advantages)  
        
        # 计算GRPO的surrogate loss（最大化期望奖励）  
        policy_ratio = torch.exp(log_probs - ref_log_probs.detach())  
        
        # 应用比率裁剪（如果启用）  
        if self.mu < 1.0:  
            clipped_ratio = torch.clamp(policy_ratio, 1 - self.mu, 1 + self.mu)  
            surrogate1 = policy_ratio * all_advantages  
            surrogate2 = clipped_ratio * all_advantages  
            surrogate_loss = -torch.min(surrogate1, surrogate2).mean()  
        else:  
            surrogate_loss = -(policy_ratio * all_advantages).mean()  
        
        # 计算KL散度惩罚项  
        kl_loss = (log_probs - ref_log_probs.detach()).mean()  
        
        # 总损失  
        total_loss = surrogate_loss + self.kl_coef * kl_loss  
        
        return total_loss  

    def _get_batch_logprobs(self, logits, labels, non_padding_mask):  
        """计算批次的对数概率  
        
        对于每个序列，计算所有(非pad)token的条件对数概率之和  
        """  
        # 对logits应用log_softmax  
        log_probs = F.log_softmax(logits, dim=-1)  
        
        # 收集label对应的对数概率  
        # 将labels移动到与log_probs相同的设备  
        labels = labels.to(log_probs.device)  
        
        # 获取每个位置上标签对应的对数概率  
        token_log_probs = torch.gather(log_probs[:, :-1], -1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)  
        
        # 应用非填充掩码，计算每个序列的对数概率之和  
        # 确保掩码尺寸正确  
        seq_log_probs = (token_log_probs * non_padding_mask[:, 1:]).sum(dim=1)  
        
        return seq_log_probs  



        

class GRPOTrainerWrapper:  
    def __init__(  
        self,  
        output_dir: str = GRPO_MODEL_PATH,  
        dataset_name_or_path: str = DPO_DATA_PATH,  
        sft_dataset_name_or_path: str = DATA_PATH,
        cached_dataset_name_or_path:str = CACHED_GRPO_DATA_PATH,
        model_name: str = MODEL_PATH,  
        is_ds: bool = True,  
        ds_config_path: Optional[str] = None,  
        is_peft: bool = False,  
        peft_config: Optional[LoraConfig] = None,  
        is_quantized: bool = False,  
        bnb_config: Optional[BitsAndBytesConfig] = None,  
        max_seq_length: int = 1024,  
        grpo_config: Optional[GRPOConfig] = None  
    ):  
        self.output_dir = output_dir
        self.dataset_name_or_path = dataset_name_or_path  
        self.sft_dataset_name_or_path = sft_dataset_name_or_path
        self.cached_dataset_name_or_path = cached_dataset_name_or_path
        self.max_seq_length = max_seq_length  
        self.is_quantized = is_quantized
        
        # GRPO配置  
        self.grpo_config = grpo_config or GRPOConfig()  
        self.beta = self.grpo_config.beta  
        self.group_size = self.grpo_config.group_size  
        self.mu = self.grpo_config.mu  
        self.kl_coef = self.grpo_config.kl_coef  
        self.scale_rewards = self.grpo_config.scale_rewards  

        # 初始化模型和tokenizer  
        self.model, self.tokenizer = self._init_model_and_tokenizer(  
            model_name, is_quantized, bnb_config  
        )  
        
        # 创建参考模型（使用深拷贝确保参数独立）  
        self.ref_model = self._clone_model(self.model)  
        self.ref_model.eval()  # 设置为评估模式  
        self.ref_model.requires_grad_(False)  # 冻结参数  
        
        # 应用LoRA  
        if is_peft:  
            self.peft_config = peft_config or self._default_lora_config()  
            self.model = get_peft_model(self.model, self.peft_config)  

        # 准备数据集  
        self.dataset = self._prepare_dataset()  

        # 配置训练参数  
        self.training_args = TrainingArguments(  
            output_dir=output_dir,  
            deepspeed=ds_config_path if is_ds else None,  
            per_device_train_batch_size=4,  
            gradient_accumulation_steps=2,  
            learning_rate=2e-5,  
            bf16=True,  
            logging_steps=10,  
            save_steps=500,  
            remove_unused_columns=False,  
            optim="adamw_torch",  
            max_grad_norm=0.3,  
            num_train_epochs=3  
        )  

        self.trainer = GRPOTrainer(  
            ref_model = self.ref_model,
            tokenizer = self.tokenizer,
            max_seq_length= self.max_seq_length,
            model=self.model,  
            args=self.training_args,  
            train_dataset=self.dataset,  
            eval_dataset = self.eval_dataset,
            data_collator=self.grpo_collator,  
            # compute_metrics=self._compute_metrics,
        )  

    def _init_model_and_tokenizer(self, model_name, is_quantized, bnb_config):  
        bnb_config = bnb_config or BitsAndBytesConfig(  
            load_in_4bit=True,  
            bnb_4bit_quant_type="nf4",  
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,  
        ) if is_quantized else None  

        tokenizer = AutoTokenizer.from_pretrained(model_name)  
        tokenizer.pad_token = tokenizer.eos_token  

        model = Qwen2ForCausalLM.from_pretrained(  
            model_name,  
            quantization_config=bnb_config,  
            device_map="auto",  
            trust_remote_code=True  
        )  
        return model, tokenizer  

    def _clone_model(self, model):  
        """创建模型的深拷贝"""  
        model_copy = copy.deepcopy(model)  
        return model_copy  

    def _default_lora_config(self):  
        return LoraConfig(  
            r=64,  
            lora_alpha=16,  
            lora_dropout=0.05,  
            target_modules=["q_proj", "v_proj"],  
            bias="none",  
            task_type="CAUSAL_LM"  
        )  

    def _prepare_dataset(self, train_size = 1000, eval_size = 500):  
        '''
        从头开始预处理数据
        
        train数据使用的是DPO数据集, 每个样本包含 (prompt, chosen, rejected) 3个字段
        eval数据使用的是 SFT 数据集 （travel-qa），每个样本包含 (Question, Answer) 两个字段
        '''
        train_dataset = load_dataset(self.dataset_name_or_path, split='train').select(range(train_size))  
        eval_dataset = load_dataset(self.sft_dataset_name_or_path)
        
        if "validation" in eval_dataset:
            eval_dataset = load_dataset(self.sft_dataset_name_or_path, split='validation').select(range(eval_size)) 
        elif "test" in eval_dataset:
            eval_dataset = load_dataset(self.sft_dataset_name_or_path, split='test').select(range(eval_size))
        else:
            eval_dataset = load_dataset(self.sft_dataset_name_or_path, split='train').select(range(eval_size))
            
        train_dataset = train_dataset.filter(self._data_filter)  
        eval_dataset = eval_dataset.filter(self._data_filter)  

        
        train_data = train_dataset.map(  
            self._tokenize_train_function,  
            batched=True,  
            num_proc=1,  
            remove_columns=train_dataset.column_names  
        )  
        
        val_data = eval_dataset.map(
            self._tokenize_eval_function,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names
        )
        
        tokenized_data = DatasetDict({
            'train': train_data,
            'validation': val_data
        })
        
        # 保存预处理好的数据集到本地
        if not os.path.exists(CACHED_GRPO_DATA_PATH):
            os.makedirs(CACHED_GRPO_DATA_PATH, exist_ok=True)
        tokenized_data.save_to_disk(CACHED_GRPO_DATA_PATH)
        
        
        train_data.set_format(type="torch")
        val_data.set_format(type="torch")
        
        return CustomGRPODataset(train_data), QADataset(val_data)

    def _data_filter(self, sample):  
        return all([sample["prompt"], sample["answer"]]) and \
               len(sample["prompt"]) <= 512 and \
               len(sample["answer"]) <= 1024  
               
               

    def _tokenize_train_function(self, samples):  
        """  
        处理训练数据集（DPO格式，包含prompt, chosen, rejected）  
        返回适用于GRPO训练的格式  
        """  
        
        # chosen_input_ids 实际上已经被添加到 batch["input_ids"] 中，这部分逻辑是正确的。
        
        batch = {  
            "input_ids": [],            # 选择回答的完整输入  
            "attention_mask": [],       # 选择回答的注意力掩码  
            "rejected_input_ids": [],   # 拒绝回答的完整输入  
            "rejected_attention_mask": [] # 拒绝回答的注意力掩码  
        }  
        # batch = {"input_ids": [], "attention_mask": [], "response_ids": [], "group_labels": []}  
        
        for prompt, chosen, rejected in zip(samples["prompt"], samples["chosen"], samples["rejected"]):  
            # 处理chosen回答  
            chosen_prompt_tokens = self.tokenizer(  
                f"Question: {prompt}\nAnswer:",   
                max_length=self.max_seq_length // 2,  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            chosen_response_tokens = self.tokenizer(  
                chosen,  
                max_length=self.max_seq_length // 2,  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            # 合并输入ID和attention mask (chosen)  
            chosen_input_ids = torch.cat([  
                chosen_prompt_tokens["input_ids"][0],   
                chosen_response_tokens["input_ids"][0][1:]  # 去掉response的BOS token  
            ])  
            chosen_attention_mask = torch.cat([  
                chosen_prompt_tokens["attention_mask"][0],   
                chosen_response_tokens["attention_mask"][0][1:]  
            ])  
            
            # 裁剪到最大长度  
            if len(chosen_input_ids) > self.max_seq_length:  
                chosen_input_ids = chosen_input_ids[:self.max_seq_length]  
                chosen_attention_mask = chosen_attention_mask[:self.max_seq_length]  
            
            # 处理rejected回答  
            rejected_prompt_tokens = self.tokenizer(  
                f"Question: {prompt}\nAnswer:",   
                max_length=self.max_seq_length // 2,  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            rejected_response_tokens = self.tokenizer(  
                rejected,  
                max_length=self.max_seq_length // 2,  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            # 合并输入ID和attention mask (rejected)  
            rejected_input_ids = torch.cat([  
                rejected_prompt_tokens["input_ids"][0],   
                rejected_response_tokens["input_ids"][0][1:]  # 去掉response的BOS token  
            ])  
            rejected_attention_mask = torch.cat([  
                rejected_prompt_tokens["attention_mask"][0],   
                rejected_response_tokens["attention_mask"][0][1:]  
            ])  
            
            # 裁剪到最大长度  
            if len(rejected_input_ids) > self.max_seq_length:  
                rejected_input_ids = rejected_input_ids[:self.max_seq_length]  
                rejected_attention_mask = rejected_attention_mask[:self.max_seq_length]  
            
            # 添加到批次  
            batch["input_ids"].append(chosen_input_ids)  
            batch["attention_mask"].append(chosen_attention_mask)  
            batch["rejected_input_ids"].append(rejected_input_ids)  
            batch["rejected_attention_mask"].append(rejected_attention_mask)  
        
        # 使用PyTorch内置的pad_sequence或您已有的填充方法  
        if hasattr(self, "_pad_sequences"):  
            batch["input_ids"] = self._pad_sequences(batch["input_ids"])  
            batch["attention_mask"] = self._pad_sequences(batch["attention_mask"])  
            batch["rejected_input_ids"] = self._pad_sequences(batch["rejected_input_ids"])  
            batch["rejected_attention_mask"] = self._pad_sequences(batch["rejected_attention_mask"])  
        else:  
            # 使用PyTorch内置的padding  
            from torch.nn.utils.rnn import pad_sequence  
            batch["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)  
            batch["attention_mask"] = pad_sequence(batch["attention_mask"], batch_first=True, padding_value=0)  
            batch["rejected_input_ids"] = pad_sequence(batch["rejected_input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)  
            batch["rejected_attention_mask"] = pad_sequence(batch["rejected_attention_mask"], batch_first=True, padding_value=0)  
        
        return batch  
        
    
    def _tokenize_eval_function(self, samples):  
        """  
        处理评估数据集（travel_qa格式，包含Question和Response/Answer）  
        返回适用于评估的格式  
        """  
        batch = {  
            "input_ids": [],  
            "attention_mask": [],  
            "labels": []  
        }  
        
        # 确定问题和答案字段名  
        question_field = "Question" if "Question" in samples else "question"  
        answer_field = "Response" if "Response" in samples else "Answer" if "Answer" in samples else "answer"  
        
        for question, answer in zip(samples[question_field], samples[answer_field]):  
            # 对问题进行编码  
            question_tokens = self.tokenizer(  
                f"Question: {question}\nAnswer:",  
                max_length=self.max_seq_length // 2,  # 预留一半长度给回答  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            # 对回答进行编码  
            answer_tokens = self.tokenizer(  
                answer,  
                max_length=self.max_seq_length // 2,  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            # 合并输入ID和attention mask  
            input_ids = torch.cat([  
                question_tokens["input_ids"][0],  
                answer_tokens["input_ids"][0][1:]  # 去掉answer的BOS token  
            ])  
            attention_mask = torch.cat([  
                question_tokens["attention_mask"][0],  
                answer_tokens["attention_mask"][0][1:]  
            ])  
            
            # 创建标签 (-100表示不计算损失的token，即提问部分)  
            labels = torch.cat([  
                torch.full_like(question_tokens["input_ids"][0], -100),  
                answer_tokens["input_ids"][0][1:]  # 去掉answer的BOS token  
            ])  
            
            # 裁剪到最大长度  
            if len(input_ids) > self.max_seq_length:  
                input_ids = input_ids[:self.max_seq_length]  
                attention_mask = attention_mask[:self.max_seq_length]  
                labels = labels[:self.max_seq_length]  
            
            batch["input_ids"].append(input_ids)  
            batch["attention_mask"].append(attention_mask)  
            batch["labels"].append(labels)  
        
        # 填充成相同长度的张量  
        batch["input_ids"] = self._pad_sequences(batch["input_ids"], max_seq_length = self.max_seq_length)  
        batch["attention_mask"] = self._pad_sequences(batch["attention_mask"], max_seq_length = self.max_seq_length)  
        batch["labels"] = self._pad_sequences(batch["labels"], padding_value=-100, max_seq_length = self.max_seq_length)  
        
        return batch  
    
    
    def _pad_sequences(self, sequences, padding_value=0, max_seq_length = 1024):  
        """  
        将不同长度的序列填充到相同长度  
        """  
        # max_length = max(len(seq) for seq in sequences)  
        max_length = max_seq_length
        padded_sequences = []  
        
        for seq in sequences:  
            padding_length = max_length - len(seq)  
            padded_seq = torch.cat([  
                seq,   
                torch.full((padding_length,), padding_value, dtype=seq.dtype)  
            ])  
            padded_sequences.append(padded_seq)  
        
        return torch.stack(padded_sequences)  
    
    

    def grpo_collator(self, features):  
        """GRPO数据整理函数，处理批次数据"""  
        # 填充输入序列  
        input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)  
        attention_mask = pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0)  
        
        # 如果有response_ids，填充它们  
        if features[0]["response_ids"] is not None:  
            response_ids = pad_sequence([f["response_ids"] for f in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)  
        else:  
            response_ids = None  
            
        # 如果有group_labels，收集它们  
        if features[0]["group_labels"] is not None:  
            group_labels = torch.stack([f["group_labels"] for f in features])  
        else:  
            group_labels = None  
            
        batch = {  
            "input_ids": input_ids,  
            "attention_mask": attention_mask,  
        }  
        
        if response_ids is not None:  
            batch["response_ids"] = response_ids  
            
        if group_labels is not None:  
            batch["group_labels"] = group_labels  
            
        return batch  


    def train(self):  
        self.trainer.train()  

    def save_model(self):  
        save_path = os.path.join(self.output_dir, "qwen2_grpo")  
        self.trainer.save_model(save_path)  
        self.tokenizer.save_pretrained(save_path)  