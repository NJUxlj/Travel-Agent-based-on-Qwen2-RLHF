from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class ModelUtils:
    @staticmethod
    def load_base_model(model_name: str = "Qwen/Qwen2-7B", 
                       device_map: str = "auto") -> tuple:
        """
        加载基础模型和分词器
        
        Args:
            model_name: 模型名称或路径
            device_map: 设备映射策略
        
        Returns:
            tuple: (model, tokenizer)
        """
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 确保分词器具有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.float16  # 使用float16以节省显存
        )
        
        return model, tokenizer
    
    @staticmethod
    def prepare_model_for_lora(
        model: AutoModelForCausalLM,
        lora_config: Optional[Dict] = None
    ) -> AutoModelForCausalLM:
        """
        为模型添加LoRA配置
        
        Args:
            model: 基础模型
            lora_config: LoRA配置参数
        
        Returns:
            添加了LoRA的模型
        """
        default_config = {
            "r": 8,  # LoRA秩
            "lora_alpha": 32,  # LoRA alpha参数
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]  # 需要训练的模块
        }
        
        # 使用用户配置更新默认配置
        if lora_config:
            default_config.update(lora_config)
        
        # 创建LoRA配置
        peft_config = LoraConfig(**default_config)
        
        # 获取PEFT模型
        model = get_peft_model(model, peft_config)
        
        return model
    
    @staticmethod
    def generate_response(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        生成模型响应
        
        Args:
            model: 模型
            tokenizer: 分词器
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
        
        Returns:
            str: 生成的响应文本
        """
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除prompt部分
        response = response[len(prompt):]
        
        return response.strip()