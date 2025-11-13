from src.tools.prompt_template import MyPromptTemplate
from src.tools.tool_executor import ToolDispatcher
from typing import Dict, List, Optional, Tuple
from src.data.data_processor import CrossWOZProcessor
import asyncio
import os
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from typing_extensions import Never

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    Case,
    Default,
)


from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from typing import Dict, List, Optional, Tuple, Literal


from dotenv import load_dotenv

load_dotenv()

'''
建议检查Pydantic版本兼容性，推荐使用：

pip install pydantic>=2.5.0  

'''

from datasets import load_dataset
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction    # pyright: ignore[reportMissingImports]
import re
import os
import torch
from src.configs.config import RAG_DATA_PATH, SFT_MODEL_PATH, EMBEDDING_MODEL_PATH


ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")

class MyAgent():
    """
    这是一个专门优化旅行规划的Agent，它内部包含以下几个子agent：
    1. 文章分析agent：分析文章的结构和主题，给出优化建议。
    2. 语言优化agent：检查文章中的语法错误和用词不当之处，给出优化建议。
    3. 内容丰富agent：根据文章主题，提出可以进一步扩展和丰富的内容点或改进建议。
    """
    
    def __init__(
        self, 
        use_api:bool = True, 
        travel_agent=None,
        use_rag = False,
        use_langchain_agent = False
        ):
        self.use_api = use_api
        self.use_rag = use_rag
        self.use_langchain_agent = use_langchain_agent

        
        if not self.use_api:
            self.agent = travel_agent
            
        
        self.base_template = "你是一个拥有10年导游经验的旅行规划助手，你非常熟悉海外旅游业务(包括但不限于：机票、酒店、美食、交通、景点、本地向导)，"
        
    def call_local_model(self, prompt):
    
        # 加载本地模型
        model:AutoModelForCausalLM|GenerationMixin = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH)
        model = model.to("cuda")
        
        tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
        
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
        
    
    #https://open.bigmodel.cn/ 注册获取APIKey
    #https://www.bigmodel.cn/dev/api/normal-model/glm-4  API 文档
    def call_api_model(self, prompt)->str:
        client = ZhipuAI(api_key=ZHIPU_API_KEY) # 填写您自己的APIKey
        response = client.chat.completions.create(
            model="glm-4-flash",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response.choices[0].message.content
        return response_text
    
    
    def plan_structure_agent(self, travel_plan_response:str):
        """向大模型提问进行travel plan分析, 返回当前期旅行计划的结构"""
        prompt_analysis = self.base_template + f"请整理，分析并输出以下旅行计划的结构，包括关键节点，关键路线+交通方式，关键操作：{travel_plan_response}"
        # 调用大模型接口，假设返回的结果是一个字典，包含结构和主题信息
        
        if self.use_api:
            plan_structure_result = self.call_api_model(prompt_analysis)
        else:
            plan_structure_result = self.call_local_model(prompt_analysis)
            
        return plan_structure_result
    
    
    def language_optimization_agent(self, travel_plan_response, plan_structure_result):
        # 根据{路劲规划结构}构建提示词
        prompt_language = self.base_template + f"请检查下面的旅游路线规划中的语法错误和用词不当之处，并提出优化建议。建议要尽量简练，不超过200字。\n\n旅游规划结构：{plan_structure_result}\n\n旅游规划内容{travel_plan_response}"
        language_optimization_suggestions = self.call_api_model(prompt_language)
        return language_optimization_suggestions

    def content_enrichment_agent(self, travel_plan_response, plan_structure_result):
        # 根据文章分析结果构建提示词
        prompt_content = self.base_template + f"请阅读下面这个旅游规划方案，根据所提供的旅游规划结构，为该规划提出可以进一步扩展和丰富的内容点或改进建议，比如添加周边推荐、修改错误数据，更新最短路径等。建议要尽量简练，不超过100字。\n\n旅游规划结构：{plan_structure_result}\n\n旅游规划内容：{travel_plan_response}"
        content_enrichment_suggestions = self.call_api_model(prompt_content)
        return content_enrichment_suggestions

    def readability_evaluation_agent(self, travel_plan_response, plan_structure_result):
        # 根据文章分析结果构建提示词
        prompt_readability = self.base_template + f"请阅读下面这个旅游规划方案，根据所提供的旅游规划结构评估该规划的可读性，包括段落长度、句子复杂度等，提出一些有助于规划实施的改进建议。建议要尽量简练，不超过100字。\n\n旅游规划结构：{plan_structure_result}\n\n旅游规划内容：{travel_plan_response}"
        readability_evaluation_result = self.call_api_model(prompt_readability)
        return readability_evaluation_result

    def comprehensive_optimization_agent(self, travel_plan_response, plan_structure_result, language_optimization_suggestions, content_enrichment_suggestions, readability_evaluation_result):
        # 合并结果的逻辑可以是将各个部分的建议整理成一个结构化的文档
        final_optimization_plan = self.base_template + f"请阅读下面的旅游路线规划，以及若干个负责专项优化的agent给出的改进建议，重写这个旅游规划方案，提升规划内容的整体质量。\n\n旅游规划原文:{travel_plan_response}\n\n旅游规划结构：{plan_structure_result}\n\n语言优化建议：{language_optimization_suggestions}\n\n内容丰富建议：{content_enrichment_suggestions}\n\n可读改进建议：{readability_evaluation_result}。\n\n优化后的旅游规划方案："
        final_optimization_result = self.call_api_model(final_optimization_plan)
        return final_optimization_result
    
    
    
    def get_final_plan(self, travel_plan_response):
        '''
        使用 Chain 来顺序调用多个agent来完成任务
        '''
        structure = self.plan_structure_agent(travel_plan_response)
        language_optimization_suggestions = self.language_optimization_agent(travel_plan_response, structure)
        content_enrichment_suggestions = self.content_enrichment_agent(travel_plan_response, structure)
        readability_evaluation_result = self.readability_evaluation_agent(travel_plan_response, structure)
        final_result = self.comprehensive_optimization_agent(travel_plan_response, structure, language_optimization_suggestions, content_enrichment_suggestions, readability_evaluation_result)
        
        return final_result
    
    
    



class AgentWithLangChain(): 
    
    
    def __init__(
        self,
        chain_type:Literal["stuff", "map_reduce", "refine"] = "stuff",
        ):
        pass




        
        
if __name__ == '__main__':
    pass
    
        
    
        