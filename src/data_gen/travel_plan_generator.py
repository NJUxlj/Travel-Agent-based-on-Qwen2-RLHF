import os, sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Annotated, Union
import json
from pprint import pprint
from datetime import datetime
from dataclasses import dataclass
from typing_extensions import Literal, Never
from pydantic import BaseModel
import asyncio 
from agent_framework.openai import OpenAIChatClient
from agent_framework._tools import BaseTool


from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Role,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowEvent,
    WorkflowOutputEvent,
    executor,
    WorkflowViz,
    Executor,
    handler
)

from dotenv import load_dotenv
load_dotenv()

'''
借助 AlphaZero 的思想， 利用 self-play 策略来实现输出的旅游规划文本的质量提升
'''

sys.path.append(Path(__file__).parent.parent)
from configs.llm_config import OpenAIChatConfig



class TravelPlanGenerator:
    def __init__(
        self, 
        llm_config:OpenAIChatConfig = OpenAIChatConfig()):
        self.llm_config = llm_config


    


