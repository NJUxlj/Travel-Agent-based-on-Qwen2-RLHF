from typing import Optional
import os

import serpapi
# pydantic_v1: 用于更方便地构建和验证与语言模型交互的数据结构。
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool




class HotelsInput(BaseModel):
    pass