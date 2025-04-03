import torch  
from torch.utils.data import Dataset  
import torch.nn.functional as F  
from transformers import (  
    AutoTokenizer,  
    TrainingArguments,  
    Trainer,  
    TrainerCallback,  
    Qwen2ForCausalLM  
)  
from peft import LoraConfig, get_peft_model  
import os  
import numpy as np  
from datasets import load_dataset, load_from_disk  

class PPODataset(Dataset):  
    def __init__(self, tokenized_data):  
        self.data = tokenized_data  
        
    def __len__(self):  
        return len(self.data["input_ids"])  
    
    def __getitem__(self, idx):  
        # 返回单个数据样本，不再包含预先计算的advantages和returns  
        # 这些值将在训练过程中动态计算  
        return {  
            "input_ids": self.data["input_ids"][idx],  
            "attention_mask": self.data["attention_mask"][idx],  
            "response_ids": self.data["response_ids"][idx],  
            "old_log_probs": self.data["old_log_probs"][idx],  
        }  

class PPOTrainer:  
    """  
    PPO训练器类  
    实现PPO算法的核心训练逻辑  
    """ 
    def __init__(  
        self,  
        output_dir: str,  
        dataset_path: str,  
        model_name: str = "qwen/Qwen2-7B",  # 基础模型名称
        clip_epsilon: float = 0.2,     # PPO裁剪参数，限制策略更新幅度
        gamma: float = 0.99,           # 折扣因子，控制未来奖励的重要性  
        gae_lambda: float = 0.95,      # GAE λ参数，平衡偏差和方差 
        ppo_epochs: int = 3,            # 每批数据的PPO更新次数 
        lr: float = 3e-5,              # 学习率 
        max_seq_length: int = 1024,  
        is_peft: bool = True,  
        peft_config: LoraConfig = None  
    ):  
        self.output_dir = output_dir  
        self.clip_epsilon = clip_epsilon  
        self.gamma = gamma  
        self.gae_lambda = gae_lambda  
        self.ppo_epochs = ppo_epochs  

        # 初始化模型和tokenizer  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        self.tokenizer.pad_token = self.tokenizer.eos_token  
        
        # 创建主策略模型（Actor）  - 这是我们要优化的模型 
        self.model = Qwen2ForCausalLM.from_pretrained(  
            model_name,  
            device_map="auto",  
            trust_remote_code=True  
        )  
        
        # 创建参考策略模型 - 用于KL散度计算，防止策略偏离太远  
        # 类似于RLHF中的SFT模型，作为行为基准  
        self.ref_model = Qwen2ForCausalLM.from_pretrained(  
            model_name,  
            device_map="auto",  
            trust_remote_code=True  
        )  
        # 冻结参考模型  
        for param in self.ref_model.parameters():  
            param.requires_grad = False  
        
        # 创建价值头网络（Critic）- 用于预测状态值函数V(s)  
        # 在PPO中，Actor-Critic架构是标准做法   
        self.value_head = torch.nn.Linear(  
            self.model.config.hidden_size,    # 输入维度：模型隐藏状态大小 
            1                                 # 输出维度：单一值，表示状态价值  
        ).to(self.model.device)  
        
        # 应用LoRA  
        if is_peft:  
            self.peft_config = peft_config or self._default_lora_config()  
            self.model = get_peft_model(self.model, self.peft_config)  

        # 准备训练数据集  
        self.dataset = self._prepare_dataset(dataset_path, max_seq_length)  

        # 配置优化器 - 同时优化策略模型和价值头   
        self.optimizer = torch.optim.AdamW(  
            list(self.model.parameters()) + list(self.value_head.parameters()),   
            lr=lr,  
            weight_decay=0.01   # 权重衰减，防止过拟合  
        )  

    def _default_lora_config(self):  
        return LoraConfig(  
            r=64,  
            lora_alpha=16,  
            lora_dropout=0.05,  
            target_modules=["q_proj", "v_proj"],  
            bias="none",  
            task_type="CAUSAL_LM"  
        )  

    def _prepare_dataset(self, dataset_path, max_length):  
        """数据预处理流程"""  
        dataset = load_dataset(dataset_path, split="train")  
        
        def process_fn(samples):  
            batch = {"input_ids": [], "attention_mask": [],  
                    "response_ids": [], "old_log_probs": []}  
            
            for prompt, chosen in zip(samples["prompt"], samples["chosen"]):  
                # 生成完整prompt  
                full_prompt = f"Instruction: {prompt}\nResponse: {chosen}"  
                
                # Tokenize输入  
                tokens = self.tokenizer(  
                    full_prompt,  
                    max_length=max_length,  
                    padding="max_length",  
                    truncation=True,  
                    return_tensors="pt"  
                )  
                
                # 计算旧策略概率  
                with torch.no_grad():  
                    logits = self.model(**tokens).logits  
                    log_probs = F.log_softmax(logits, dim=-1)  
                    old_log_probs = torch.gather(  
                        log_probs, -1, tokens["input_ids"].unsqueeze(-1)  
                    ).squeeze(-1)  
                
                batch["input_ids"].append(tokens["input_ids"][0])  
                batch["attention_mask"].append(tokens["attention_mask"][0])  
                batch["response_ids"].append(tokens["input_ids"][0])  
                batch["old_log_probs"].append(old_log_probs[0])  
            
            return batch  
        
        tokenized_data = dataset.map(  
            process_fn,  
            batched=True,  
            num_proc=4,  
            remove_columns=dataset.column_names  
        )  
        return PPODataset(tokenized_data)  

    def ppo_collator(self, features):  
        return {  
            "input_ids": torch.stack([f["input_ids"] for f in features]),  
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),  
            "response_ids": torch.stack([f["response_ids"] for f in features]),  
            "old_log_probs": torch.stack([f["old_log_probs"] for f in features])  
        }  
        
    def compute_rewards(self, model_outputs, response_ids, attention_mask, kl_coef=0.1):  
        """计算奖励，包括外部奖励和KL惩罚"""  
        # 这里示例使用简单奖励函数（可以替换为实际奖励函数）  
        # 例如，使用生成文本的长度作为正向奖励  
        rewards = []  
        
        # 计算新策略生成的概率  
        logits = model_outputs.logits  
        log_probs = F.log_softmax(logits, dim=-1)  
        new_log_probs = torch.gather(  
            log_probs, -1, response_ids.unsqueeze(-1)  
        ).squeeze(-1)  
        
        # 计算参考模型的概率（用于KL散度）  
        with torch.no_grad():  
            ref_outputs = self.ref_model(  
                input_ids=response_ids,  
                attention_mask=attention_mask  
            )  
            ref_logits = ref_outputs.logits  
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)  
            ref_log_probs = torch.gather(  
                ref_log_probs, -1, response_ids.unsqueeze(-1)  
            ).squeeze(-1)  
            
        # 计算KL散度  
        kl = (new_log_probs - ref_log_probs) * attention_mask  
        
        # 生成简单奖励（例如基于令牌数量），这里仅作为示例  
        # 实际使用中应该替换为自定义奖励函数  
        base_rewards = torch.sum(attention_mask, dim=-1).float() * 0.1  
        
        # 应用KL惩罚  
        rewards = base_rewards - kl_coef * torch.sum(kl, dim=-1)  
        
        return rewards, new_log_probs, kl  
        
    def compute_gae(self, rewards, values, masks, gamma=0.99, lam=0.95):  
        """计算广义优势估计(GAE)"""  
        advantages = torch.zeros_like(rewards)  
        last_gae_lam = 0  
        
        # 反向遍历序列以计算GAE  
        for t in reversed(range(len(rewards))):  
            if t == len(rewards) - 1:  
                next_value = 0  
            else:  
                next_value = values[t + 1]  
                
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]  
            last_gae_lam = delta + gamma * lam * masks[t] * last_gae_lam  
            advantages[t] = last_gae_lam  
            
        # 计算回报（returns = advantages + values）  
        returns = advantages + values  
        
        # 标准化优势  
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  
        
        return advantages, returns  

    def compute_loss(self, model, inputs):  
        # 前向传播获取logits  
        outputs = model(  
            input_ids=inputs["input_ids"],  
            attention_mask=inputs["attention_mask"]  
        )  
        
        # 计算奖励和KL散度  
        rewards, new_log_probs, kl = self.compute_rewards(  
            outputs,   
            inputs["response_ids"],  
            inputs["attention_mask"]  
        )  
        
        # 获取隐藏状态用于价值估计  
        hidden_states = outputs.hidden_states[-1]  # 获取最后一层的隐藏状态  
        
        # 使用价值头计算每个token的价值  
        values = self.value_head(hidden_states).squeeze(-1)  
        
        # 计算蒙版（用于忽略padding）  
        masks = inputs["attention_mask"]  
        
        # 计算GAE优势和回报  
        advantages, returns = self.compute_gae(  
            rewards, values, masks,   
            gamma=self.gamma, lam=self.gae_lambda  
        )  
        
        # 计算新策略概率  
        log_probs = F.log_softmax(outputs.logits, dim=-1)  
        new_log_probs = torch.gather(  
            log_probs, -1, inputs["response_ids"].unsqueeze(-1)  
        ).squeeze(-1)  
        
        # 计算概率比率  
        ratio = torch.exp(new_log_probs - inputs["old_log_probs"])  
        
        # 计算策略损失 (clipped PPO objective)  
        policy_loss = -torch.min(  
            ratio * advantages,  
            torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages  
        ).mean()  
        
        # 计算价值函数损失  
        value_loss = F.mse_loss(values, returns)  
        
        # 计算熵奖励（增加探索性）  
        entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1).mean()  
        
        # 总损失  
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy  
        
        return total_loss  

    def train(self):  
        """自定义训练循环"""  
        train_args = TrainingArguments(  
            output_dir=self.output_dir,  
            per_device_train_batch_size=4,  
            gradient_accumulation_steps=2,  
            learning_rate=3e-5,  
            logging_steps=10,  
            save_steps=500,  
            remove_unused_columns=False,  
            optim="adamw_torch",  
            max_grad_norm=0.5  
        )  
        
        trainer = Trainer(  
            model=self.model,  
            args=train_args,  
            train_dataset=self.dataset,  
            data_collator=self.ppo_collator,  
            compute_metrics=None,  
            compute_loss=self.compute_loss  
        )  
        
        # PPO多轮优化  
        for _ in range(self.ppo_epochs):  
            trainer.train()  
            
    def save_model(self):  
        save_path = os.path.join(self.output_dir, "qwen2_ppo")  
        self.model.save_pretrained(save_path)  
        self.tokenizer.save_pretrained(save_path)  