from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from typing import List, Tuple, Union, Dict, Optional



# 构建dataset
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer: AutoTokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer:AutoTokenizer = tokenizer
        
        self.final_prompts = []
        
        for prompt in prompts:
            if apply_chat_template:
                content = [{"role":"user", "content":prompt}]    
                prompt = self.tokenizer.apply_chat_template(content, tokenize = False, add_generation_prompt = True)
            else:
                prompt = self.tokenizer.bos_token + prompt
                
            self.final_prompts.append(prompt)
        
    def __len__(self):
        return len(self.prompts)

    def __getitem(self, idx):
        return self.final_prompts[idx]
    
    
    



# 价值（评论家）模型，用于预测每一步（生成token）的动作产生的收益，使用演员模型进行初始化，并外加一个回归头，输出shape为：(batch_size, seq_len， 1)
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
        
    def forward(self, input_ids, attention_mask, num_actions):
        '''
        num_asctions: 模型新推理了多少token
        '''
        
        hidden_states = self.base_model.forward(
            input_ids,
            attention_mask = attention_mask
        ).last_hidden_state  # shape = (batch_size, seq_len, hidden_size)
        
        value_model_output = self.value_head.forward(hidden_states)  # shape = (batch_size, seq_len, 1)
        
        values = value_model_output.squeeze(-1)[:, -num_actions:] # 只取策略做出的actions（response）的部分
        
        return values
    
    






def compute_policy_loss(log_probs:torch.Tensor, old_log_probs:torch.Tensor, advantages, action_mask = None, clip_eps = 0.2):
    '''
    advantage: 
        优势函数，用于衡量当前策略相对于旧策略的优势。
        advantage.shape = (batch_size, seq_len)
        
    log_probs:
        策略模型输出的logits
        log_probs.shape = (batch_size, seq_len)
        
    old_log_probs:
        旧策略模型输出的logits
        old_log_probs.shape = (batch_size, seq_len)
        
    action_mask:
        用于指示哪些动作是有效的，哪些动作是无效的。
        action_mask.shape = (batch_size, seq_len)
    '''
    ratio = (log_probs - old_log_probs).exp()  # A/B = exp[log(A/B)] = exp[logA - LogB] 
    
    # ratio.shape = (batch_size, seq_len)
    
    surrogate1 = ratio * advantages
    surrogate2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    
    loss = - torch.min(surrogate1, surrogate2) # shape =  (bsz, seq_len)
     
    if action_mask is None:
        return loss.mean(-1).mean()  # 先对序列维度求平均，再对批次维度求平均
    else:
        return ((loss*action_mask).sum(-1) / action_mask.sum(-1)).mean()




def compute_value_loss(values, old_values, returns, action_mask = None, clip_eps:float = None):
    '''
    计算价值模型的损失函数（Value Loss），用于优化评论家（Critic）模型。
    
    参数:
        values (torch.Tensor): 当前价值模型预测的状态价值，shape=(batch_size, seq_len), Critic模型预测的状态价值（V(s)），表示在当前策略下从状态s开始的预期回报
        old_values (torch.Tensor): 旧价值模型预测的状态价值，shape=(batch_size, seq_len)
        returns (torch.Tensor): 实际观察到的回报（return），shape=(batch_size, seq_len), 实际观察到的折扣回报（G_t）
        action_mask (torch.Tensor, optional): 动作掩码，指示哪些位置是有效动作，shape=(batch_size, seq_len)
        clip_eps (float, optional): 用于限制价值更新的裁剪系数。如果为None则不使用裁剪
        
    返回:
        torch.Tensor: 计算得到的价值损失标量值
        
    损失函数：
        loss = (values - returns) ** 2
        等价于  L = (V(s) - G_t)²
        
        本质：
            是价值函数（Critic）训练中的均方误差（MSE）损失计算
        
        作用：  
            1. 这是典型的时序差分（Temporal Difference）学习中的价值函数拟合
            2. 目标是最小化预测价值与实际回报之间的平方误差
            3. 通过最小化这个损失，Critic模型逐渐学会更准确地预测每个状态的期望回报
        
        其中：
            values：Critic模型预测的状态价值（V(s)），表示在当前策略下从状态s开始的预期回报
            returns：实际观察到的折扣回报（G_t），通过蒙特卡洛方法或TD(λ)等方法计算得到
        
    实现细节:
        1. 当clip_eps不为None时，使用PPO的裁剪机制防止价值函数更新过大:
           - 计算裁剪后的价值预测: values_clipped = old_values + clip(values - old_values)
           - 取裁剪和非裁剪损失中的较大值: loss = max((values_clipped-returns)^2, (values-returns)^2)
        2. 当clip_eps为None时，直接计算MSE损失: loss = (values - returns)^2
        3. 如果有action_mask，则只计算有效位置的损失
        4. 最终对序列维度和批次维度求平均得到标量损失值
        
    数学公式:
        clipped_loss = (values_clipped - returns)^2
        unclipped_loss = (values - returns)^2
        loss = max(clipped_loss, unclipped_loss)  # 当使用clip时
        或
        loss = unclipped_loss  # 当不使用clip时
    '''
    if clip_eps is None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surrogate1 = (values_clipped- returns)**2
        surrogate2 = (values - returns) ** 2
        
        loss = torch.max(surrogate1, surrogate2)
    
    else:
        loss = (values - returns) ** 2
    
    
    if action_mask is None:
        return loss.mean(-1).mean()

    else:
        return ((loss*action_mask).sum(-1) / action_mask.sum(-1)).mean()



    
    
    
    
    
    
class ExperienceBuffer:
    '''
    经验池
    
    作用：
        存储我们采样到的轨迹：Trajectories
    '''
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []  # 存储轨迹的列表，也就是实际上的轨迹数据集D

    
    def append(self, experiences:List["Experience"]):
        '''
        删除旧数据， 保留新数据
        
        作用：
            将experiences对象列表，转为batch这个字典列表
            
            再把batch添加到buffer里面
        '''
        batch = [{} for _ in range(len(experiences))]  
        
        keys = (
            "seqs",
            "action_log_probs",
            "values",  # Critic模型预测的状态价值（V(s)），shape = (batch_size, seq_len)
            "returns", # 实际观察到的折扣回报（G_t），shape = (batch_size, seq_len)
            "advantages",
            "attention_mask",
            "action_mask",
            "num_actions" # 模型新推理了多少token
        )
        
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
        
        self.buffer.extend(batch)
        
        
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer)-self.limit:]
            
        
        
        
    def get_batches(self, batch_size):
        '''
        随机抽取一个批次
        '''
        batch = random.sample(self.buffer, batch_size)
        return batch

    
    
    
    def clear(self):
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)
    
    
    
    def __getitem__(self, index):
        return self.buffer[index]
    
    
    


@dataclass
class Samples:
    '''
    存储策略模型的输出
    '''
    seqs:torch.Tensor
    attention_mask:Optional[torch.LongTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    
    
@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None




def compute_approx_kl(
    log_probs: torch.Tensor,  # 策略模型输出的logits
    ref_log_probs: torch.Tensor, # 参考模型输出的logits
    action_mask: Optional[torch.Tensor] = None,
):
    log_ratio = log_probs.float() - ref_log_probs.float()  # log(A/B) = logA - logB
    
    if action_mask  is not None:
        log_ratio = log_ratio * action_mask
    
    return log_ratio





def get_advantages_and_returns(
    values: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
    lambd: float
    ):
    '''
    ### 原理：
        # A(t) = R(t) + gam*V(t+1) - V(t)
        # gae:A(t) = R(t) + gam*V(t+1) - V(t) + gam*lam*A(t+1)
        # 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0,  则A(T) = R(T) - V(T), 得出A(T)
        # A(T-1) = R(T-1) + gam*V(T) - V(T-1) + gam*lam*A(T) 知道A(T)可计算A(T-1) 依次类推
        # returns(t) = A(t) + V(t) = = R(t) + gam * (V(t+1) + lam * A(t+1))
    '''
    
    



def compute_rewards():
    
    
    steps = 0
    for episode in range(episodes):
        pass




def generate_experiences(samples_list):
    
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()
    
    experiences  =[]
    
    
    
    for sample in samples_list:
        pass
    
    
    
    
    
    
    
    
    
    return experiences






@dataclass
class BufferItem:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    
    
    
    
    


def collate_fn(batch):
    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []







    
    

    
    
def train_step(experience, steps):
    
    actor_model.train()
    optimizer_actor.zero_grad()
    
    
    logits = actor_model(
            sequences,
            attention_mask = attention_mask
        ).logits


def train():
    # 初始化经验池, 经验池就是论文中的 轨迹数据集 D (Trajectories)
    buffer = ExperienceBuffer(limit=100)   # 先取 100 个 prompt， yong来生成轨迹
    
    steps = 0
    
    
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
             # 生成样本（获取模型推理结果）, 也叫做 生成轨迹数据集 D
            samples = generate_samples(rand_prompts, actor_model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size)
            
             # 生成经验（获取优势、奖励、回报等）
            experiences = generate_experiences(samples) # 计算轨迹数据集D中的每条轨迹的奖励、回报、优势等
            
            buffer.append(experiences)



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 一共迭代多少轮
    episodes = 3
    # 生成一次经验 (i.e., Trajectories dataset)，训练的轮数
    max_epochs = 5
    # 一次从提示词数据集中取多少条数据用于生成经验 
    rollout_batch_size = 8
    # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
    micro_rollout_batch_size = 2
    # 一个提示词生成多少个样本
    n_samples_per_prompt = 2
    
    
    # 轨迹数据集 D 的大小 =  rollout_batch_size *  n_samples_per_prompt
    
    
    # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_new_tokens = 50
    # 最大长度
    max_length = 256
    # 实际训练的batch_size大小，一次取多少条数据用于更新参数
    micro_train_batch_size = 2     # mini-batch
    # 记录日志
    writer = SummaryWriter('./runs')
    # 策略模型
    actor_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
    # 参考模型
    ref_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
    # 奖励模型
    reward_model = AutoModelForSequenceClassification.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2').to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct')
    reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    # 价值模型
    critic_model = Critic(actor_model.base_model).to(device)
    
    # 初始化优化器
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)
    
    # 填充方式为左填充
    actor_tokenizer.padding_side = 'left'
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id
    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)
   
    train()