
from prompt_template import PromptTemplate
from typing import Dict, List, Optional, Tuple
from src.models.model import TravelAgent


class RAG():
    def __init__(
        self, 
        agent: TravelAgent,
        use_langchain = False,
        use_prompt_template = True,
        use_db = False
        ):
        self.use_langchain = use_langchain
        self.use_prompt_template = use_prompt_template
        self.agent = agent
    
    def construct_prompt(self, query):
        pass
    
    
    def chat(self):
        pass
    
    def search_embedding_db(self):
        pass
    
    
    
    def rag_chat(self,):
        pass
    
    
    
    
    
    def summarize_results(self, results:Dict)->str:
        """将原始结果转换为自然语言摘要"""  
        summaries = []  
        for item in results.get("items", []):  
            summaries.append(f"标题：{item['title']}\n摘要：{item['snippet']}")  
        return "\n\n".join(summaries) 
    
    
    
    
        
        
    
        