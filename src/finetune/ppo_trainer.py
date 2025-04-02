import torch  
from torch.utils.data import Dataset  
from transformers import (  
    AutoTokenizer,  
    TrainingArguments,  
    Trainer,  
    TrainerCallback,  
    Qwen2ForCausalLM  
)  
from peft import LoraConfig, get_peft_model  
import os  


from datasets import load_dataset, load_from_disk

class PPODataset(Dataset):  
    def __init__(self, tokenized_data):  
        self.data = tokenized_data  
        
    def __len__(self):  
        return len(self.data["input_ids"])  
    
    def __getitem__(self, idx):  
        return {  
            "input_ids": self.data["input_ids"][idx],  
            "attention_mask": self.data["attention_mask"][idx],  
            "response_ids": self.data["response_ids"][idx],  
            "old_log_probs": self.data["old_log_probs"][idx],  
            "advantages": self.data["advantages"][idx],  
            "returns": self.data["returns"][idx]  
        }  

class PPOTrainer:  
    def __init__(  
        self,  
        output_dir: str,  
        dataset_path: str,  
        model_name: str = "qwen/Qwen2-7B",  
        clip_epsilon: float = 0.2,  
        gamma: float = 0.99,  
        gae_lambda: float = 0.95,  
        ppo_epochs: int = 3,  
        lr: float = 3e-5,  
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
        
        self.model = Qwen2ForCausalLM.from_pretrained(  
            model_name,  
            device_map="auto",  
            trust_remote_code=True  
        )  
        
        # 应用LoRA  
        if is_peft:  
            self.peft_config = peft_config or self._default_lora_config()  
            self.model = get_peft_model(self.model, self.peft_config)  

        # 准备数据集  
        self.dataset = self._prepare_dataset(dataset_path, max_seq_length)  

        # 配置优化器  
        self.optimizer = torch.optim.AdamW(  
            self.model.parameters(),   
            lr=lr,  
            weight_decay=0.01  
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
                    "response_ids": [], "old_log_probs": [],  
                    "advantages": [], "returns": []}  
            
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
                
                # 计算旧策略概率（需要预先生成）  
                with torch.no_grad():  
                    logits = self.model(**tokens).logits  
                    log_probs = F.log_softmax(logits, dim=-1)  
                    old_log_probs = torch.gather(  
                        log_probs, -1, tokens["input_ids"].unsqueeze(-1)  
                    ).squeeze(-1)  
                
                # 计算advantages和returns（需要预先生成）  
                # 这里假设数据已包含相关字段  
                batch["input_ids"].append(tokens["input_ids"][0])  
                batch["attention_mask"].append(tokens["attention_mask"][0])  
                batch["response_ids"].append(tokens["input_ids"][0])  
                batch["old_log_probs"].append(old_log_probs[0])  
                batch["advantages"].append(torch.randn(max_length))  # 示例数据  
                batch["returns"].append(torch.randn(max_length))     # 示例数据  
            
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
            "old_log_probs": torch.stack([f["old_log_probs"] for f in features]),  
            "advantages": torch.stack([f["advantages"] for f in features]),  
            "returns": torch.stack([f["returns"] for f in features])  
        }  

    def compute_loss(self, model, inputs):  
        # 前向传播  
        outputs = model(  
            input_ids=inputs["input_ids"],  
            attention_mask=inputs["attention_mask"]  
        )  
        
        # 计算新策略概率  
        log_probs = F.log_softmax(outputs.logits, dim=-1)  
        new_log_probs = torch.gather(  
            log_probs, -1, inputs["response_ids"].unsqueeze(-1)  
        ).squeeze(-1)  
        
        # 计算概率比率  
        ratio = torch.exp(new_log_probs - inputs["old_log_probs"])  
        
        # 计算策略损失  
        advantages = inputs["advantages"]  
        policy_loss = -torch.min(  
            ratio * advantages,  
            torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages  
        ).mean()  
        
        # 计算价值函数损失  
        values = model.get_output_embeddings()(outputs.logits)  
        value_loss = F.mse_loss(values.squeeze(), inputs["returns"]).mean()  
        
        # 计算熵奖励  
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
            compute_metrics=None  
        )  
        
        # PPO多轮优化  
        for _ in range(self.ppo_epochs):  
            trainer.train()  
            
    def save_model(self):  
        save_path = os.path.join(self.output_dir, "qwen2_ppo")  
        self.model.save_pretrained(save_path)  
        self.tokenizer.save_pretrained(save_path)