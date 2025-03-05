
from src.agents.prompt_template import MyPromptTemplate
from src.agents.tools import ToolDispatcher
from typing import Dict, List, Optional, Tuple
from src.models.model import TravelAgent
from src.data.data_processor import CrossWOZProcessor




from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain  
from langchain.memory.buffer import ConversationBufferMemory  
from langchain_community.vectorstores import Chroma      # pip install langchain-chroma  pip install langchain_community
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnablePassthrough  
from langchain_core.output_parsers import StrOutputParser  
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain_core.prompts import PromptTemplate

from langchain.graphs import Neo4jGraph  
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
# from langchain_experimental.graph_transformers import ChainMap     # pip install langchain_experimental


'''
建议检查Pydantic版本兼容性，推荐使用：

pip install pydantic>=2.5.0  

'''

from datasets import load_dataset
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction  
import re
import torch
from zhipuai import ZhipuAI 

from src.configs.config import RAG_DATA_PATH, SFT_MODEL_PATH, EMBEDDING_MODEL_PATH





class LocalEmbeddingFunction(EmbeddingFunction):  
    """本地嵌入模型适配器"""  
    def __init__(self, model_name: str = EMBEDDING_MODEL_PATH):  
        self.embedder = HuggingFaceEmbeddings(  
            model_name=model_name,  
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},  
            encode_kwargs={'normalize_embeddings': True}  
        )  

    def __call__(self, texts: List[str]) -> List[List[float]]:  
        return self.embedder.embed_documents(texts)  





class RAG():
    def __init__(
        self, 
        agent: TravelAgent,
        dataset_name_or_path:str = RAG_DATA_PATH,
        embedding_model_name_or_path:str = EMBEDDING_MODEL_PATH,
        use_langchain = False,
        use_prompt_template = True,
        use_db = True
        ):
        self.use_langchain = use_langchain
        self.use_prompt_template = use_prompt_template
        self.use_db = use_db
        self.agent = agent
        
        if use_db:
            self.embedding_fn = LocalEmbeddingFunction(model_name=embedding_model_name_or_path)
            self.chroma_client = chromadb.Client()
            # self.chroma_client = chromadb.PersistentClient(path = "local dir")
            
            self.collection = self.chroma_client.create_collection(
                name="my_collection",
                embedding_function=self.embedding_fn,
                metadata={
                    "hnsw:space":"cosine",
                    "embedding_model": embedding_model_name_or_path
                })
            self.dataset =  load_dataset(dataset_name_or_path, split="train").select(range(1000))
            
            self.embeddings = HuggingFaceEmbeddings(   # langchain 专用
                model_name=EMBEDDING_MODEL_PATH
            )  
            
            # 加载数据集时自动生成嵌入  
            self._initialize_database() 
        
        
        if self.use_prompt_template:
            self.prompt_template = MyPromptTemplate()
            self.dispatcher = ToolDispatcher()
    
    
    # def parse_db(self):
    #     assert self.use_db, "The embedding database is not initialized."
    #     result = []
        
    #     for sample in self.dataset:
    #         result.append(sample["history"])
            
    #     return result
    
    
    
    def _initialize_database(self):  
        """使用本地嵌入模型初始化数据库"""  
        batch_size = 100  
        for i in range(0, len(self.dataset), batch_size):  
            batch = self.dataset[i:i+batch_size]  
            documents = [item["history"] for item in batch]  
            metadatas = [{"source": "crosswoz"}] * len(documents)  
            ids = [str(idx) for idx in range(i, i+len(documents))]  
            
            self.collection.add(  
                documents=documents,  
                metadatas=metadatas,  
                ids=ids  
            )  
        
    
    
    def query_db(self, user_query:str, n_results=5)->List[str]:
        assert self.use_db, "The embedding database is not initialized."
        
        # 使用本地模型生成查询嵌入  
        query_embedding = self.embedding_fn([user_query])[0]
        
        # corpus = self.parse_db()
        # ids = [f"id{i+1}" for i in range(len(corpus))]
        
        # self.collection.add(
        #     documents = corpus,
        #     # metadatas = [{"source": "my_source"}, {"source": "my_source"}],
        #     ids = ids
        # )
        
        results = self.collection.query(
            query_embeddings = [query_embedding],
            # query_texts= [user_query],
            n_results = n_results,
            # where = {"metadata_field": "is_equal_to_this"},
            # where_document = {"$contains": "search_string"}
            include=["documents", "distances"] 
        )
        
        # 添加相似度阈值过滤  
        filtered = [  
            doc for doc, dist in zip(results["documents"][0], results["distances"][0])  
            if 1 - dist > 0.7  # cosine距离转相似度  
        ]  
        
        return filtered
        
    
    
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
        """完全基于LangChain API实现的增强版对话"""  
        print("============ LangChain RAG Chat 启动 ===========")  
        
        
        # 初始化LangChain组件  
        memory = ConversationBufferMemory(  
            return_messages=True,   
            output_key="answer",  
            memory_key="chat_history"  
        )  
        
        # 创建检索器（使用已初始化的Chroma集合）  
        retriever = Chroma(  
            client=self.chroma_client,  
            collection_name="my_collection",  
            embedding_function=self.embeddings  # 需补充实际embedding模型  
        ).as_retriever(search_kwargs={"k": 5})   # 设置每次检索返回最相关的5个结果
        
        # 构建工具调用链  
        tool_chain = (  
            RunnablePassthrough.assign(  
                context=lambda x: retriever.get_relevant_documents(x["question"])  
            )  
            | self._build_tool_prompt()  
            | self.agent.model  # 假设已适配LangChain接口  
            | StrOutputParser()  
        )  
        
        
        # 启动对话循环  
        while True:  
            user_input = input("User: ")  
            if user_input.lower() == "exit":  
                print("Goodbye!")  
                break  
            
            response = tool_chain.invoke({  
                "question": user_input,  
                "chat_history": memory.load_memory_variables({})["chat_history"]  
            })  
            
            # 解析工具调用  
            tool_result = self._process_langchain_response(response)  
            memory.save_context({"input": user_input}, {"output": tool_result})  
            
            print(f"Assistant: {tool_result}")  
            print("=============================================")  
    
    
    def _build_tool_prompt(self):  
        """构建集成工具和context的提示模板""" 
        
        return ChatPromptTemplate.from_template("""  
            结合以下上下文和工具调用结果回答问题：  
            
            上下文信息：  
            {context}  
            
            历史对话：  
            {chat_history}  
            
            用户问题：{question}  
            
            请按以下格式响应：  
            {tool_format}  
            """).partial(  
                tool_format=self.prompt_template.generate_prompt("", "")  
            )  
            
    def _process_langchain_response(self, response: str) -> str:  
        """处理LangChain输出并执行工具调用"""  
        try:  
            # 添加工具调用频率限制  
            if len(re.findall(r"<工具调用>", response)) > 10:  
                return "检测到过多工具调用，请简化您的问题"
            
            # 解析工具调用字符串  
            tool_call = re.search(r"<工具调用>(.*?)</工具调用>", response, re.DOTALL)  
            if not tool_call:  
                return response  

            # 执行工具调用  
            result = self.dispatcher.execute(tool_call.group(1).strip())  
            
            # 数据库查询不应该在这里
            db_result = self.query_db(tool_call.group(1)) if self.use_db else ""  
            
            return f"{response}\n\n工具执行结果：{result}\n数据库匹配结果：{db_result}"  
        
        except Exception as e:  
            return f"Error processing response: {str(e)}"  
    
    
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
    
    
    
    
class GraphRAG(RAG):
    def __init__(
        self, 
        agent: TravelAgent,
        dataset_name_or_path:str = RAG_DATA_PATH,
        embedding_model_name_or_path:str = EMBEDDING_MODEL_PATH,
        use_langchain = False,
        use_prompt_template = True,
        use_db = True
    ):
        super().__init__(
            agent=agent,
            dataset_name_or_path = dataset_name_or_path,
            embedding_model_name_or_path= embedding_model_name_or_path,
            use_langchain = use_langchain,
            use_prompt_template = use_prompt_template,
            use_db = use_db
        )
    
    def _initialize_knowledge_graph(self) -> Neo4jGraph:  
        """构建知识图谱"""  
        # 连接到Neo4j（示例配置，需根据实际修改）  
        graph = Neo4jGraph(  
            url="bolt://localhost:7687",  
            username="neo4j",  
            password="password"  
        )  
        
        # 从文档中提取实体关系  
        query = """  
        UNWIND $documents AS doc  
        CALL apoc.nlp.gcp.entities.analyze({  
            text: doc.text,  
            key: $apiKey,  
            types: ["PERSON","LOCATION","ORGANIZATION"]  
        }) YIELD value  
        UNWIND value.entities AS entity  
        MERGE (e:Entity {name: entity.name})  
        SET e.type = entity.type  
        WITH e, doc  
        MERGE (d:Document {id: doc.id})  
        MERGE (d)-[:CONTAINS]->(e)  
        """  
        
        # 批量处理文档（示例）  
        documents = [{"id": str(i), "text": d["history"]} for i, d in enumerate(self.dataset)]  
        graph.query(query, params={"documents": documents, "apiKey": "your-gcp-key"})  
        
        return graph  
    
    
    
    def _build_graph_prompt(self) -> PromptTemplate:  
        """构建图谱增强的提示模板"""  
        return PromptTemplate.from_template("""  
            结合知识图谱和文本上下文回答下列问题：  
            
            知识图谱路径：  
            {graph_paths}  
            
            相关文本：  
            {context}  
            
            历史对话：  
            {chat_history}  
            
            问题：{question}  
            
            请按照以下要求回答：  
            1. 明确提及相关实体  
            2. 说明实体间的关系  
            3. 保持回答简洁专业  
            """)  
        
        
    
    def graph_rag_chat(self):  
        """基于GraphRAG的对话实现"""  
        from langchain.memory import ConversationBufferMemory  
        from langchain.chains import ConversationalRetrievalChain  
        
        # 初始化组件  
        memory = ConversationBufferMemory(  
            memory_key="chat_history",  
            return_messages=True,  
            output_key="answer"  
        )  
        
        # 创建混合检索器  
        class GraphEnhancedRetriever:  
            def __init__(self, vector_retriever, graph):  
                self.vector_retriever = vector_retriever  
                self.graph = graph  
                
            def get_relevant_documents(self, query: str) -> List[Dict]:  
                # 向量检索  
                vector_docs = self.vector_retriever.get_relevant_documents(query)  
                
                # 图谱检索  
                graph_query = f"""  
                MATCH path=(e1)-[r]->(e2)  
                WHERE e1.name CONTAINS '{query}' OR e2.name CONTAINS '{query}'  
                RETURN path LIMIT 5  
                """  
                graph_paths = self.graph.query(graph_query)  
                
                return {  
                    "vector_docs": vector_docs,  
                    "graph_paths": graph_paths  
                }  

        # 初始化检索器  
        vector_retriever = Chroma(  
            client=self.chroma_client,  
            collection_name="my_collection"  
        ).as_retriever()  
        
        hybrid_retriever = GraphEnhancedRetriever(vector_retriever, self.graph)  
        
        # 创建对话链  
        qa_chain = ConversationalRetrievalChain.from_llm(  
            llm=self.agent.llm,  
            retriever=hybrid_retriever,  
            memory=memory,  
            combine_docs_chain_kwargs={  
                "prompt": self._build_graph_prompt(),  
                "document_prompt": PromptTemplate(  
                    input_variables=["page_content"],  
                    template="{page_content}"  
                )  
            },  
            get_chat_history=lambda h: "\n".join([f"User:{u}\nAssistant:{a}" for u, a in h])  
        )  
        
        # 启动对话循环  
        print("========== GraphRAG对话系统启动 ==========")  
        while True:  
            try:  
                query = input("用户: ")  
                if query.lower() in ["exit", "quit"]:  
                    break  
                
                result = qa_chain({"question": query})  
                print(f"助手: {result['answer']}")  
                print("\n知识图谱路径:")  
                for path in result["graph_paths"]:  
                    print(f"- {path['start_node']['name']} → {path['relationship']} → {path['end_node']['name']}")  
                print("=====================================")  
                
            except KeyboardInterrupt:  
                break  







    
# 辅助函数 
def visualize_knowledge_graph(graph: Neo4jGraph):  
    """可视化知识图谱（示例）"""  
    query = """  
    MATCH (n)-[r]->(m)  
    RETURN n.name AS source,   
           type(r) AS relationship,   
           m.name AS target  
    LIMIT 50  
    """  
    return graph.query(query)  
        
        
        
        
if __name__ == '__main__':
    pass
    
        
    
        