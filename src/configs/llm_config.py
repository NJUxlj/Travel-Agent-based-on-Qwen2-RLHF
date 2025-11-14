from dataclasses import dataclass, field
from pydantic import BaseModel
from dotenv import load_dotenv
import os, sys

from pydantic_core.core_schema import dataclass_field


load_dotenv()  # load environment variables from .env



@dataclass
class OpenAIChatConfig:
    model_name: str = "zhipu"

    def __post_init__(self):
        """根据model_name从环境变量读取对应的配置"""
        if self.model_name == "zhipu":
            self.model_id = os.getenv("ZHIPU_MODEL_ID")
            self.api_key = os.getenv("ZHIPU_API_KEY")
            self.endpoint = os.getenv("ZHIPU_ENDPOINT")
        elif self.model_name == "qwen":
            self.model_id = os.getenv("QWEN_MODEL_ID")
            self.api_key = os.getenv("QWEN_API_KEY")
            self.endpoint = os.getenv("QWEN_ENDPOINT")
        elif self.model_name == "gpt5":
            self.model_id = os.getenv("GPT5_MODEL_ID")
            self.api_key = os.getenv("GPT5_API_KEY")
            self.endpoint = os.getenv("GPT5_ENDPOINT")
        elif self.model_name == "openai":
            self.model_id = os.getenv("OPENAI_MODEL_ID", "gpt-3.5-turbo")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        # 验证必要的配置不为空
        if not self.api_key:
            raise ValueError(f"API key is required for {self.model_name}")
        if not self.model_id:
            raise ValueError(f"Model ID is required for {self.model_name}")
        if not self.endpoint:
            raise ValueError(f"Endpoint is required for {self.model_name}")
