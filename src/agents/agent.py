
from src.agents.prompt_template import MyPromptTemplate
from src.agents.tools import ToolDispatcher
from typing import Dict, List, Optional, Tuple
from src.models.model import TravelAgent
from src.data.data_processor import CrossWOZProcessor

from datasets import load_dataset
import chromadb

from src.configs.config import RAG_DATA_PATH, SFT_MODEL_PATH


class RAG():
    def __init__(
        self, 
        agent: TravelAgent,
        dataset_name_or_path:str = RAG_DATA_PATH,
        use_langchain = False,
        use_prompt_template = True,
        use_db = True
        ):
        self.use_langchain = use_langchain
        self.use_prompt_template = use_prompt_template
        self.use_db = use_db
        self.agent = agent
        
        if use_db:
            self.chroma_client = chromadb.Client()
            # self.chroma_client = chromadb.PersistentClient(path = "local dir")
            
            self.collection = self.chroma_client.create_collection(
                name="my_collection",
                metadata={
                    "hnsw:space":"cosine"
                })
            self.dataset =  load_dataset(dataset_name_or_path, split="train").select(range(1000))
        
        
        if self.use_prompt_template:
            self.prompt_template = MyPromptTemplate()
            self.dispatcher = ToolDispatcher()
    
    
    def parse_db(self):
        assert self.use_db, "The embedding database is not initialized."
        result = []
        
        for sample in self.dataset:
            result.append(sample["history"])
            
        return result
        
    
    
    def query_db(self, user_query:str, n_results=5)->List[str]:
        assert self.use_db, "The embedding database is not initialized."
        corpus = self.parse_db()
        
        ids = [f"id{i+1}" for i in range(len(corpus))]
        
        self.collection.add(
            documents = corpus,
            # metadatas = [{"source": "my_source"}, {"source": "my_source"}],
            ids = ids
        )
        
        results = self.collection.query(
            # query_embeddings = [[11.1, 12.1, 13.1], [1.1, 2.3, 3.2]],
            query_texts= [user_query],
            n_results = n_results,
            # where = {"metadata_field": "is_equal_to_this"},
            # where_document = {"$contains": "search_string"}
        )
        
        return results["documents"][0]
        
    
    
    def chat(self):
        '''
        simple chat without RAG
        '''
        self.agent.chat()
    
    def rag_chat(self,):
        history = [("You are a very help AI assistant who can help me plan a wonderful trip for a vacation",
                    "OK, I know you want to have a good travel plan and I will answer your questions about the traveling spot and search for the best plan about the traveling route and hotel.")]

        print("\n\n\n=============================")
        print("============ Welcome to the TravelAgent Chat! Type 'exit' to stop chatting. ==========")  
        while True:  
            user_input = input(f"User: ")  
            if user_input.lower() == 'exit':  
                print("Goodbye!")  
                break  
            
            prompt = self.prompt_template.generate_prompt(
                user_input,
                "\n".join([f"User:{user}\nSystem:{sys}" for user, sys in history])
                )
            
            # formatted_history = " ".join([f"User: {user}\nSystem: {sys}\n" for user, sys in history])

            
            tool_call_str = self.agent.generate_response(
                prompt,
                max_length=2048
                )  
            
            print(" ================ 模型返回的包含工具调用的response =======================")
            print(tool_call_str)
            print("===========================================")
            
            # 工具调用
            raw_result = self.dispatcher.execute(tool_call_str)
            
            # 数据库匹配
            db_result = self.query_db(user_input) if self.use_db else ""
            db_result = "\n".join(db_result)
            
            final_response = tool_call_str + f"""
            
            工具调用结果是：
            {raw_result}
            
            数据库查询的结果是：
            {db_result}
            """
            print("=============== 集成所有的工具信息后的prompt ===============")
            print(final_response)
            print("=====================================================")
            
            travel_plan = self.get_travel_plan(final_response, max_length=256)
            # summary = self.summarize_results(final_response)
            # 总结
            print(f"TravelAgent: {travel_plan}")  
            print(" ======================================= ")
            
            history.append((user_input, travel_plan))
    
    
    
    def langchain_rag_chat(self):
        pass
    
    
    
    
    
    def summarize_results(self, results:Dict)->str:
        """将原始结果转换为自然语言摘要"""  
        summaries = []  
        for item in results.get("items", []):  
            summaries.append(f"标题：{item['title']}\n摘要：{item['snippet']}")  
        return "\n\n".join(summaries) 
    
    
    def get_travel_plan(self, query:str, max_length = 512):
        SYS_PROMPT = "你是一个旅行助手，可以帮助我规划一条合适的旅游路线. 基于下面的信息: {query}, 请你帮我规划一条合理的路由路线. 你返回的路线比如用列表的形式组织，并且清晰，简洁."

        
        response = self.agent.generate_response(SYS_PROMPT, max_length=max_length)
        
        
        return response
    
    
    
        
        
        
        
if __name__ == '__main__':
    pass
    
        
    
        