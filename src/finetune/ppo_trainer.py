import torch  
import torch.nn as nn
from torch.utils.data import Dataset  
import torch.nn.functional as F  
from transformers import (  
    AutoTokenizer,  
    TrainingArguments,  
    Trainer,  
    TrainerCallback,  
    AutoModelForSequenceClassification,
    DebertaV2Model,
    
    DebertaV2Config
)  

from datasets import DatasetDict


from tqdm import tqdm


from src.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from src.configs.config import (
    REWARD_MODEL_PATH, 
    MODEL_PATH, 
    SFT_MODEL_PATH, 
    PPO_MODEL_PATH, 
    DPO_DATA_PATH,
    CACHED_DPO_DATA_PATH,
    CACHED_PPO_DATA_PATH,
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
        
        
        
        
class Critic(nn.Module):
    '''
    # 价值（评论家）模型：
    #   - 用于预测每一步（生成token）的动作产生的收益，使用演员模型进行初始化，并外加一个回归头，输出shape为：(batch_size, seq_len， 1)
    #   - 通过策略模型初始化得到
    
    '''
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # 可以看做一个 encoder
        self.base_model.eval()  # 冻结骨干部分
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(
            self, 
            input_ids = None, 
            inputs_embeds = None,
            attention_mask = None, 
            num_actions=None
            ):
        '''
        num_actions: 模型新推理了多少token
        
        return values: shape = (batch_size, seq_len)
        '''
        
        assert input_ids is not None or inputs_embeds is not None, f"one of the input_ids and inputs_embeds has to be not None"
        
        
        if input_ids is not None:
            hidden_states = self.base_model.forward(
                input_ids = input_ids,
                attention_mask = attention_mask
            ).last_hidden_state  # shape = (batch_size, seq_len, hidden_size)
        else:
            hidden_states = self.base_model.forward(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask
            ).last_hidden_state
        
        value_model_output = self.value_head.forward(hidden_states)  # shape = (batch_size, seq_len, 1)
        
        # values = value_model_output.squeeze(-1)[:, -num_actions:] # 只取策略做出的actions（response）的部分
        
        values = value_model_output.squeeze(-1) # 只取策略做出的actions（response）的部分
        
        
        return values
    
        

class PPOTrainer:  
    """  
    PPO训练器类  
    实现PPO算法的核心训练逻辑  
    """ 
    def __init__(  
        self,  
        output_dir: str = PPO_MODEL_PATH,  
        dataset_path: str = DPO_DATA_PATH,  
        cached_data_path:str = CACHED_PPO_DATA_PATH,
        model_name: str = MODEL_PATH,  # 基础模型名称
        reward_model_name:str = REWARD_MODEL_PATH,
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
        
        self.dataset_path = dataset_path
        self.cached_data_path = cached_data_path
        self.max_seq_length = max_seq_length
        
        self.clip_epsilon = clip_epsilon  
        self.gamma = gamma  
        self.gae_lambda = gae_lambda  
        self.ppo_epochs = ppo_epochs  

        # 初始化模型和tokenizer  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        self.tokenizer.pad_token = self.tokenizer.eos_token  
        
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        
        # 创建主策略模型（Actor）  - 这是我们要优化的模型 
        self.model = Qwen2ForCausalLM.from_pretrained(  
            model_name,  
            device_map="auto",  
            trust_remote_code=True  
        )  
        
        self.device  =self.model.device
        
        # 创建参考策略模型 - 用于KL散度计算，防止策略偏离太远  
        # 类似于RLHF中的SFT模型，作为行为基准  
        self.ref_model = Qwen2ForCausalLM.from_pretrained(  
            model_name,  
            # device_map="auto",  
            trust_remote_code=True  
        ).to(self.device)

            
            
            
        # 创建奖励模型    
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            trust_remote_code=True
        ).to(self.device)
        
        
        # 创建价值头网络（Critic）- 用于预测状态值函数V(s)  
        # 在PPO中，Actor-Critic架构是标准做法   
        # self.value_head = torch.nn.Linear(  
        #     self.model.config.hidden_size,    # 输入维度：模型隐藏状态大小 
        #     1                                 # 输出维度：单一值，表示状态价值  
        # ).to(self.model.device)  
        
        self.critic_model = Critic(self.model.base_model).to(self.device)
        
        # 应用LoRA  
        if is_peft:  
            self.peft_config = peft_config or self._default_lora_config()  
            self.model = get_peft_model(self.model, self.peft_config)  

        # 准备训练数据集  
        self.dataset, self.eval_dataset = self._load_cached_dataset(self.cached_data_path)  
        

        # 配置优化器 - 同时优化策略模型和价值头   
        
        # [方案1， 可以用， 但是有点奇怪]
        # self.optimizer = torch.optim.AdamW(  
        #     list(self.model.parameters()) + list(self.value_head.parameters()),   
        #     lr=lr,  
        #     weight_decay=0.01   # 权重衰减，防止过拟合  
        # )  
        
        # 【方案2， 分别为 actor和critic创建优化器】
        self.optimizer_actor = torch.optim.Adam(self.model.parameters(), lr=0.00005)
        self.optimizer_critic = torch.optim.Adam(self.critic_model.parameters(), lr=0.00005)
        

    def _default_lora_config(self):  
        return LoraConfig(  
            r=32,  
            lora_alpha=16,  
            lora_dropout=0.05,  
            target_modules=["q_proj", "k_proj" ,"v_proj"],  
            bias="none",  
            task_type="CAUSAL_LM"  
        )  
        
        
    def _load_cached_dataset(self, dataset_path=CACHED_PPO_DATA_PATH):
        if not os.path.exists(dataset_path):
            os.makedirs(CACHED_PPO_DATA_PATH, exist_ok=True)
        try:
            tokenized_data = load_from_disk(dataset_path)
            print("从缓存加载DPO数据集成功~~~")
            # tokenized_data.set_format(type="torch")
            train_data = tokenized_data['train']
            eval_data = tokenized_data['validation']
            return PPODataset(train_data), PPODataset(eval_data)
        except Exception as e:
            print(f"加载缓存的DPO数据集（tokenized）失败: {e}, 将重新预处理数据")
            ppo_train_data, ppo_eval_data = self._prepare_dataset(self.dataset_path, self.max_seq_length)
            return ppo_train_data, ppo_eval_data

    def _prepare_dataset(self, dataset_path, max_length):  
        """数据预处理流程"""  
        dataset = load_dataset(dataset_path)
        train_dataset = load_dataset(dataset_path, split='train').select(range(500))  
        
        if "validation" in dataset:
            eval_dataset = load_dataset(dataset_path, split='validation').select(range(500)) 
        else:
            eval_dataset = load_dataset(dataset_path, split='train').select(range(500, 1000))  
            
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
                ).to(self.device)
                
                # 计算旧策略概率  (使用多项式采样替代argmax)
                with torch.no_grad():  
                    logits = self.model.forward(**tokens).logits.to(self.device)
                    log_probs = F.log_softmax(logits, dim=-1)  
                    
                    # 多项式采样
                    probs = torch.softmax(logits, dim=-1)
                    response_ids = [] # 贪心采样， 可以改成多项式采样或者top-k，
                    for i in range(probs.size(1)):  # 遍历每个token位置
                        token_probs = probs[:, i, :]  # 获取当前token位置的概率分布 (batch_size, vocab_size)
                        sampled_ids = torch.multinomial(token_probs, num_samples=1)  # (batch_size, 1)
                        response_ids.append(sampled_ids)
                    response_ids = torch.cat(response_ids, dim=-1)  # (batch_size, seq_len)
                    
                    # 只收集生成部分的log_prob
                    old_log_probs = torch.gather(  
                        log_probs, -1, response_ids.unsqueeze(-1)    # 把整个序列的logprobs都收集起来了，而不是仅仅收集action的部分
                    ).squeeze(-1)    # shape = (batch_size, seq_len)
                
                batch["input_ids"].append(tokens["input_ids"][0])  
                batch["attention_mask"].append(tokens["attention_mask"][0])  
                batch["response_ids"].append(response_ids[0])  
                batch["old_log_probs"].append(old_log_probs[0])  
            
            return batch  
        
        train_data = train_dataset.map(  
            process_fn,  
            batched=True,  
            num_proc=1,  
            remove_columns=train_dataset.column_names  
        )  
        
        val_data = eval_dataset.map(
            process_fn,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names
        )
        
        tokenized_data = DatasetDict({
            'train': train_data,
            'validation': val_data
        })
        
        
        # 保存预处理好的数据集到本地
        if not os.path.exists(CACHED_PPO_DATA_PATH):
            os.makedirs(CACHED_PPO_DATA_PATH, exist_ok=True)
        tokenized_data.save_to_disk(CACHED_PPO_DATA_PATH)
        
        
        train_data.set_format(type="torch")
        val_data.set_format(type="torch")
        
        
        
        return PPODataset(train_data), PPODataset(val_data)

    def ppo_collator(self, features):  
        return {  
            "input_ids": torch.stack([f["input_ids"] for f in features]),  
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),  
            "response_ids": torch.stack([f["response_ids"] for f in features]),  
            "old_log_probs": torch.stack([f["old_log_probs"] for f in features])  
        }  
        
    def compute_rewards(self, model_outputs, response_ids, attention_mask, kl_coef=0.1):  
        """计算奖励，包括外部奖励和KL惩罚
        
        ###Args
        
        
        
        
        ###Return
        
        
        ### 简介
            这里示例使用简单奖励函数（可以替换为实际奖励函数）
            例如，使用生成文本的长度作为正向奖励  
        
        """  
        
        # 转换成文本
        response_text = self.tokenizer.batch_decode(response_ids, skip_special_tokens = True)
        
        reward_model_inputs  =  self.reward_tokenizer(response_text, return_tensors="pt", padding=True)
        
        reward_model_inputs = {k: v.to(self.reward_model.device) for k, v in reward_model_inputs.items()}
        # 使用reward model计算外部奖励
        with torch.no_grad():
            
            # print("reward_model_inputs['input_ids'].shape = ", reward_model_inputs['input_ids'].shape)
            # print("deberta_v2.config.max_length = ", DebertaV2Config().max_position_embeddings)
            
            reward_outputs = self.reward_model.forward(
                input_ids=reward_model_inputs['input_ids'],
                attention_mask=reward_model_inputs['attention_mask']
            )  # reward_outputs.logits.shape = (bsz, 1)
            external_rewards = reward_outputs.logits.squeeze(-1)  # 奖励模型只是对整个序列做出打分
        
        # 计算KL散度惩罚
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
        kl = (new_log_probs - ref_log_probs) * attention_mask  # shape = (bsz, seq_len)
        
        # 生成简单奖励（例如基于令牌数量），这里仅作为示例  
        # 实际使用中应该替换为自定义奖励函数  
        # base_rewards = torch.sum(attention_mask, dim=-1).float() * 0.1  
        
        # 组合奖励
        # rewards = external_rewards - kl_coef * torch.sum(kl, dim=-1)  # shape = (bsz, )
        rewards =  - kl_coef * kl
        rewards[:, -1] += external_rewards
        
        
        return rewards, new_log_probs, kl  
        
    def compute_gae(self, rewards, values, masks, gamma=0.99, lam=0.95):  
        """计算广义优势估计(GAE)
        
        rewards.shape = (bsz, )
        
        values.shape = (bsz, seq_len)
        
        mask
        
        
        """  
        advantages = torch.zeros_like(rewards)  
        last_gae_lam = 0   # A_{T+1}==0
        
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

    def compute_loss(self, actor_model:Qwen2ForCausalLM, critic_model:Critic, inputs):  
        # 前向传播获取logits  
        outputs = actor_model.forward(  
            input_ids=inputs["input_ids"],  
            attention_mask=inputs["attention_mask"] ,
            output_hidden_states=True
        )  
        
        # 计算奖励和KL散度  
        rewards, new_log_probs, kl = self.compute_rewards(  
            outputs,   
            inputs["response_ids"],  
            inputs["attention_mask"]  
        )  
        
        # rewards.shape = (bsz, )
        
        # 获取隐藏状态用于价值估计  
        # hidden_states = outputs.hidden_states[-1]  # 获取最后一层的隐藏状态  
        
        # 使用价值头计算每个token的价值  
        values = critic_model.forward(input_ids = inputs["input_ids"], attention_mask=inputs["attention_mask"])   # shape = (bsz, seq_len)
        
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
            per_device_train_batch_size=2,  
            gradient_accumulation_steps=2,  
            learning_rate=3e-5,  
            logging_steps=10,  
            save_steps=500,  
            remove_unused_columns=False,  
            optim="adamw_torch",  
            max_grad_norm=0.5  
        )  
        
        train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=train_args.per_device_train_batch_size,
            collate_fn=self.ppo_collator,
            shuffle=True
        )
        
        # 自定义的PPO算法无法使用huggingface官方的Trainer 
        
        # trainer = Trainer(  
        #     model=self.model,  
        #     args=train_args,  
        #     train_dataset=self.dataset,  
        #     data_collator=self.ppo_collator,  
        #     compute_metrics=None,  
        #     compute_loss=self.compute_loss  
        # )  
        
        best_eval_loss = float('inf')
        
        # PPO多轮优化  
        for epoch in range(self.ppo_epochs):
            progress_bar = tqdm(train_dataloader, desc=f"PPO Epoch {epoch}")
            
            for batch in progress_bar:
                
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                self.optimizer_actor.zero_grad()
                
                self.optimizer_critic.zero_grad()
                
                # 前向传播
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True  # 需要获取隐藏状态用于价值估计
                )
                
                # 计算损失
                loss = self.compute_loss(self.model, self.critic_model, batch)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), train_args.max_grad_norm)
                
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                
                # 更新进度条
                progress_bar.set_postfix(loss=loss.item())
            
            
            # 将更新后的策略模型参数复制给参考模型
            self.ref_model.load_state_dict(self.model.state_dict(), strict=False)
            
            eval_loss = self.evaluate()
            
            # 保存最佳模型
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                # self.save_model()
                print(f"New best eval loss: {best_eval_loss:.4f}")
            
            
        self.save_model() 
        
        
        
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        self.critic_model.eval()
        
        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=4,  # 评估batch_size可以小一些
            collate_fn=self.ppo_collator,
            shuffle=False
        )
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 计算损失
                loss = self.compute_loss(self.model, self.critic_model, batch)
                
                total_loss += loss.item() * len(batch["input_ids"])
                total_samples += len(batch["input_ids"])
        
        avg_loss = total_loss / total_samples
        print(f"\nEvaluation - Average Loss: {avg_loss:.4f}")
        
        self.model.train()
        self.critic_model.train()
        return avg_loss
            
    def save_model(self):  
        # save_path = os.path.join(self.output_dir, "qwen2_ppo")  
        self.model.save_pretrained(self.output_dir)  
        # self.tokenizer.save_pretrained(save_path)  
        
        
        
        
        
        




if __name__ == "__main__":
    
    
    trainer = PPOTrainer()
    
    trainer.train()