from src.agents.mem_walker import MemoryTreeNode
from src.agents.mem_walker import MemoryTreeBuilder

from src.agents.mem_walker import ChatPDFForMemWalker

from src.agents.mem_walker import Navigator


from typing import Literal, Callable, Dict, Tuple


from src.configs.config import PDF_FOLDER_PATH

import asyncio

#  input: 旅游规划路径


class RAGDispatcher():
    def __init__(self, rag_type:Literal["rag","self_rag", "corrective_rag", "mem_walker"]="mem_walker"):
        self.rag_type = rag_type

    async def dispatch(self, query:str):
        # 1. 规划路径分析
        # 2. 规划路径执行
        # 3. 规划路径总结
        
        if self.rag_type == "mem_walker":
            return await self.mem_walker(query)
        
        elif self.rag_type == "rag":
            return self.rag(query)

        elif self.rag_type == "self_rag":
            return self.self_rag(query)
        elif self.rag_type == "corrective_rag":
            return self.corrective_rag(query)
        
    def rag(self, query:str):
        pass
    
    async def mem_walker(self,query:str)->str:
        builder = MemoryTreeBuilder()
        
        pdf_reader = ChatPDFForMemWalker()
        pdf_reader.ingest_all(pdf_folder_path=PDF_FOLDER_PATH)
        
        all_chunks = pdf_reader.get_memwalker_chunks()
        root = await builder.build_tree(all_chunks, model_type="api")
        
        builder.print_memory_tree(root)
    
        navigator = Navigator(model_type="api")
        answer = await navigator.navigate(
            root, 
            query
            )
        
        
        return answer
        
        
    
    
    
    def self_rag(self, query:str):
        pass
    
    
    
    def corrective_rag(self, query:str):
        pass

