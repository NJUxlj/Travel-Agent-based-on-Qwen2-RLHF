from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter






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
        self.base_model = base_model
        self.base_model.eval()  # 冻结骨干部分
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, num_actions):
        '''
        num_asctions: 模型新推理了多少token
        '''
    
    




def collate_fn(batch):
    pass




def compute_policy_loss():
    pass




def compute_value_loss():
    pass



    
    
    
    
    
    
class ExperienceBuffer:
    '''
    经验池
    '''
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
        
        
    
    
    def append(self, experiences):
        '''
        删除旧数据， 保留新数据
        '''
        batch = [{} for _ in range(len(experiences))]
        
        keys = (
            "seqs",
            "action_log_probs"
        )
        
        for key in keys:
            pass
        
        self.buffer.extend(batch)
            
        
        
        
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
    pass





def get_advantages_and_returns():
    pass



def compute_rewards():
    
    
    steps = 0
    for episode in range(episodes):
        pass


    
    
def train_step(experience, steps):
    
    actor_model.train()
    optimizer_actor.zero_grad()
    
    
    logits = actor_model(
            sequences,
            attention_mask = attention_mask
        ).logits


def train():
    # 初始化经验池
    buffer = ExperienceBuffer(limit=100)




if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 一共迭代多少轮
    episodes = 3
    # 生成一次经验，训练的轮数
    max_epochs = 5
    # 一次从提示词数据集中取多少条数据用于生成经验
    rollout_batch_size = 8
    # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
    micro_rollout_batch_size = 2
    # 一个提示词生成多少个样本
    n_samples_per_prompt = 2
    # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_new_tokens = 50
    # 最大长度
    max_length = 256
    # 实际训练的batch_size大小，一次取多少条数据用于更新参数
    micro_train_batch_size = 2
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