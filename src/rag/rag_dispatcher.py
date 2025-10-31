from pathlib import Path
import os, sys
sys.path.append(Path(__file__).parent)
from mem_walker import MemoryTreeNode
from mem_walker import MemoryTreeBuilder
from mem_walker import ChatPDFForMemWalker
from mem_walker import Navigator
from self_rag import SelfRAG
from rag_config import RAGType
from typing import Literal, Callable, Dict, Tuple


from configs.config import PDF_FOLDER_PATH

import asyncio

#  input: 旅游规划路径


class RAGDispatcher():
    def __init__(self, rag_type:Literal["rag","self_rag", "corrective_rag", "mem_walker"]="mem_walker"):
        self.rag_type = rag_type

    async def dispatch(self, query:str):
        # 1. 规划路径分析
        # 2. 规划路径执行
        # 3. 规划路径总结
        
        if self.rag_type == RAGType.MEM_WALKER:
            return await self.mem_walker(query)
        
        elif self.rag_type == RAGType.RAG:
            return self.rag(query)

        elif self.rag_type == RAGType.SELF_RAG:
            return await self.self_rag(query)
        elif self.rag_type == RAGType.CORRECTIVE_RAG:
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
        
        
    
    
    
    async def self_rag(self, query:str):
        rag = SelfRAG(model_type="api")  
        chain = await rag.build_chain()  
        
        result = await chain.ainvoke(query)  
        print(f"最终答案：{result}")  
    
    
    
    def corrective_rag(self, query:str):
        pass

