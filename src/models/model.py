from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

import sys
sys.path.append("../../")  # 添加上级目录的上级目录到sys.path
from configs.config import MODEL_CONFIG


class TravelAgent:

    def __init__(
        self,
        model_name: str = MODEL_CONFIG['model']['name'], 
        device_map: str = "auto",
        device: str = "cuda" if torch.cuda.is_available() else "cpu", 
        lora_config: Optional[Dict] = None,
        ) -> tuple:
        """
        加载基础模型和分词器
        
        Args:
            model_name: 模型名称或路径
            device_map: 设备映射策略
        
        Returns:
            tuple: (model, tokenizer)
        """
        # 初始化基础配置  
        self.device = device  
        self.device_map = device_map
        self.model_name = model_name  
        
        # 默认LoRA配置  
        self.lora_config = {  
            "r": 8,  # LoRA秩  
            "lora_alpha": 32,  # LoRA alpha参数  
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # 需要训练的模块  
            "lora_dropout": 0.1,  
            "bias": "none",  
            "task_type": TaskType.CAUSAL_LM  
        }  
        if lora_config:  
            self.lora_config.update(lora_config)  
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 确保分词器具有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = self._init_model()
        
        
    
    def _init_model(self) -> AutoModelForCausalLM:  
        """初始化模型并应用LoRA配置"""  
        # 加载基础模型  
        model = AutoModelForCausalLM.from_pretrained(  
            self.model_name,  
            trust_remote_code=True,
            torch_dtype=torch.float16,  
            # device_map=self.device_map  # 并行训练时， 不能使用自动设备映射
        )  
        
        # 应用LoRA配置  
        peft_config = LoraConfig(  
            r=self.lora_config["r"],  
            lora_alpha=self.lora_config["lora_alpha"],  
            target_modules=self.lora_config["target_modules"],  
            lora_dropout=self.lora_config["lora_dropout"],  
            bias=self.lora_config["bias"],  
            task_type=self.lora_config["task_type"]  
        )  
        
        model = get_peft_model(model, peft_config)  
        return model  
    
    # @staticmethod
    # def prepare_model_for_lora(
    #     model: AutoModelForCausalLM,
    #     lora_config: Optional[Dict] = None
    # ) -> AutoModelForCausalLM:
    #     """
    #     为模型添加LoRA配置
        
    #     Args:
    #         model: 基础模型
    #         lora_config: LoRA配置参数
        
    #     Returns:
    #         添加了LoRA的模型
    #     """
    #     default_config = {
    #         "r": 8,  # LoRA秩
    #         "lora_alpha": 32,  # LoRA alpha参数
    #         "lora_dropout": 0.1,
    #         "bias": "none",
    #         "task_type": TaskType.CAUSAL_LM,
    #         "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]  # 需要训练的模块
    #     }
        
    #     # 使用用户配置更新默认配置
    #     if lora_config:
    #         default_config.update(lora_config)
        
    #     # 创建LoRA配置
    #     peft_config = LoraConfig(**default_config)
        
    #     # 获取PEFT模型
    #     model = get_peft_model(model, peft_config)
    #     return model
    

    def generate_response(
        self,
        prompt:str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        生成模型响应
        
        Args:

            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
        
        Returns:
            str: 生成的响应文本
        """
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除prompt部分
        response = response[len(prompt):]
        
        return response.strip()