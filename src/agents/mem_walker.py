import os
from typing import List, Optional, Dict, Literal, Tuple, Any
from langchain_core.documents import Document  
from langchain.text_splitter import CharacterTextSplitter 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.llm import LLMChain



from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings  
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.vectorstores.utils import filter_complex_metadata


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from zhipuai import ZhipuAI
import re


from src.configs.config import MODEL_PATH,EMBEDDING_MODEL_PATH_BPE, EMBEDDING_MODEL_PATH, SFT_MODEL_PATH
from src.agents.chat_pdf import ChatPDF


class MemoryBase:
    
    def __init__(self):
        pass
    async def _call_model(self, prompt: ChatPromptTemplate, inputs: Dict, model_type: Literal["api", "huggingface"]) -> str:
        if model_type == "api":
            client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
            response = client.chat.completions.create(
                model="glm-4",
                messages=[{"role": "user", "content": prompt.format(**inputs)}]
            )
            return response.choices[0].message.content
        elif model_type == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
            hf_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True, device_map="auto")
            pipe = pipeline(
                "text-generation",
                model=hf_model,
                tokenizer=tokenizer,
                max_new_tokens=200,
            )
            
            model = HuggingFacePipeline(pipeline=pipe)
            chain = LLMChain(llm=model, prompt=prompt)
            return await chain.arun(inputs)
        else:
            raise ValueError("Invalid model_type, please choose either 'api' or 'huggingface'")



class MemoryTreeNode():
    def __init__(self, content: str, level: int, children: List['MemoryTreeNode'] = None, id=None):
        self.content = content
        self.level = level
        self.children = children or []
        self.parent = None
        self.id = id
        
        
    def print_node_info(self):
        print(f"Node_id: {self.id} Node Level: {self.level}, Content[:25]: {self.content[:25]}...")
        print(f"Children Count: {len(self.children)}")
        if self.parent:
            print(f"parent.level: {self.parent.level}")
            print(f"Parent Content[:25]: {self.parent.content[:25]}...")

class MemoryTreeBuilder(MemoryBase):
    def __init__(self, chunk_size=1000, max_children=5):
        self.chunk_size = chunk_size
        self.max_children = max_children
        
    async def build_tree(self, chunks: List[Document], model_type: str) -> MemoryTreeNode:
        '''
        
        
        return Root Node
        '''

        # chunks = self._chunk_text(text)
        
        # 构建叶子节点
        leaf_nodes = await self._create_leaf_nodes(chunks, model_type)
        
        # 递归构建上层节点
        current_level = leaf_nodes
        level = 1
        while len(current_level) > 1:
            parent_nodes = []
            for i in range(0, len(current_level), self.max_children):
                children = current_level[i:i+self.max_children]
                parent_content = await self._summarize_nodes(children, model_type)
                parent_node = MemoryTreeNode(parent_content, level, children, id=f"level:{level} #num:{i}")
                for child in children:
                    child.parent = parent_node
                parent_nodes.append(parent_node)
            current_level = parent_nodes
            level += 1
        return current_level[0]

    def _chunk_text(self, text: str) -> List[Document]:
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size)  
        return text_splitter.split_text(text)  

    async def _create_leaf_nodes(self, chunks: List[Document], model_type: Literal["api", "huggingface"]) -> List[MemoryTreeNode]:
        nodes = []
        for idx, chunk in enumerate(chunks):
            
            # 封装一个大的 PromptTemplate
            leaf_summary_prompt = """  
            [TEXT OF SEGMENT] 
            {text}
            
            Summarize the above text comprehensively into a fluent passage.   
            Ensure the summary captures the main ideas and key details of the segment,   
            while remaining concise and coherent.  
            """  
            
            leaf_summary_prompt = ChatPromptTemplate.from_template(
                leaf_summary_prompt
            )
            
            summary = await self._call_model(
                prompt=leaf_summary_prompt,
                inputs={"text": chunk},
                model_type=model_type
            )
            nodes.append(MemoryTreeNode(summary, level=0, id = f"level:0 #num:{idx}"))
        return nodes

    async def _summarize_nodes(self, nodes: List[MemoryTreeNode], model_type: str) -> str:
        combined = "\n\n".join([n.content for n in nodes])
        
        parent_summary_prompt = """  
        [SUMMARIES]  
        {summaries}
        Compress the above summaries into a much shorter summary.   
        Ensure the compressed summary retains the key information from all child summaries,   
        while being concise and coherent.  
        """  
        
        
        parent_summary_prompt=ChatPromptTemplate.from_template(
                parent_summary_prompt
        )
        
        
        return await self._call_model(
            prompt=parent_summary_prompt,
            inputs={"summaries": combined},
            model_type=model_type
        )
        
        
        
    def print_memory_tree(self, node: MemoryTreeNode = None, indent: str = ""):
        """递归打印内存树结构"""
        if node is None:
            return
            
        # 打印当前节点内容
        print(f"{indent}Level {node.level}: [前50字] {node.content[:50]}...")
        
        # 递归打印子节点
        for child in node.children:
            self.print_memory_tree(child, indent + "    ")
        
        
        
        
        
        

class ParseError(Exception):  
    """解析响应时发生的格式错误"""  
    pass  

class InvalidAction(Exception):  
    """无效的动作指令"""  
    pass  









class Navigator(MemoryBase):
    def __init__(self, model_type: str = "api"):
        self.model_type = model_type
        self.working_memory = []
        
        self.max_retries = 3  
        self.error_count = 0  
        self.visited = set()  # 记录已访问节点  
        self.path_stack = []  # 路径回溯栈  
        
        # 初始化分流（triage）提示模板
        self.triage_prompt = ChatPromptTemplate.from_template("""
        The following passage(s) are the summaries of the different parts of a story.
        
        
        To answer the question: {query}
        Which of the following summary is MOST LIKELY to contain information about the answer? 
        First provide reasoning to compare the summaries before you make the decision.
        Format: Reasoning:... Action: [number]
        
        {summaries}
        
        
        Relpy with the passage number as your action. 
        You MUST choose one summary number and you should reply with the following format: 
        ################################### 
        Reasoning: ... 
        Action: 0 / 1 / 2, ... 
        ###################################
        
        
        """)
        
        self.leaf_prompt = ChatPromptTemplate.from_template("""
        Read the text in triple quotes and answer a question:
        [Working Memory] {working_memory}
        
        Main Text: {text}
        Can this answer the query? {query}
        
        
        Format: Reasoning:... Action: -1/-2 Answer: [optional] 
        
        
        If the answer CANNOT be inferred from the text above, reply with action -1. 
        If the answer CAN be inferred from the text above, reply with action -2, and also provide your reasoning, and the final answer. 
        You are ONLY allowed to reply with action -2 or -1. 
        Your should reply with the following format: 
        ################################### 
        Reasoning: ... 
        Action: -2 or -1 
        Answer: (A) ... 
        ###################################
        """)
        
        # -2 代表命中text， -1 代表不命中，要回溯

    async def navigate(self, root: MemoryTreeNode, query: str) -> str:
        current_node = root
        while True:
            if self.error_count >= self.max_retries:  
                return "no answer" 
             
            if current_node.level == 0:  # Leaf node
                response = await self._handle_leaf(current_node, query)
                
                print("Leaf Node")
                current_node.print_node_info()
                print("leaf-node-response: ", response)
                # if "Answer:" in response:
                #     return response.split("Answer:")[-1].strip()
                # else:
                #     current_node = current_node.parent # 如果找不到答案就回溯
                
                try:  
                    reasoning, action, answer = self._parse_leaf(response)  
                    if action == -2:  
                        return answer  
                    elif action == -1:  
                        # 论文3.2节：回退父节点并标记当前节点为已访问  
                        self.visited.add(current_node)  
                        if not self.path_stack:  # 根节点无法继续回溯  
                            return "no answer"  
                        current_node, sibling_nodes = self.path_stack.pop()  
                        # 寻找未访问的子节点（论文Table 1示例）  
                        next_node = self._find_unvisited_child(sibling_nodes)  
                        if next_node:  
                            current_node = next_node  
                        else:   # 所有子节点均已访问过，先让当前节点停留在父节点，下一轮再去找父节点的父节点的未访问的子节点
                            continue  # 继续回溯  
                except ParseError:  
                    self.error_count += 1  
                
            else:  # Non-leaf node
                response = await self._handle_triage(current_node, query) 
                
                print("Non-Leaf Node")
                current_node.print_node_info()
                print("non-leaf-node-response: ", response)
            
                try:
                    reasoning, action = self._parse_triage(response)
                    self._save_work_memory(current_node, action)

                    # if current_node in self.visited:  
                    #     # 论文4.3节：已访问节点直接回溯  
                    #     continue  
                    
                    # 记录当前节点和所有子节点（论文3.2节导航机制）  
                    self.path_stack.append( (current_node, current_node.children) )  
                    
                    # response = await self._handle_triage(current_node, query)  
                    # try:  
                    # reasoning, action = self._parse_triage(response)  
                    chosen_node = current_node.children[action]  
                    if chosen_node in self.visited:  # 已经访问过了，需要回退
                        raise InvalidAction  
                    current_node = chosen_node  # 访问选中的子节点
                except (InvalidAction):  
                    # 回溯到父节点
                    current_node, sibling_nodes = self.path_stack.pop()  
                    print(f"回溯：current_node = {current_node}")
                    current_node = self._find_unvisited_child(sibling_nodes)  
                
                except ParseError:
                    # 论文4.4节：错误恢复机制  
                    self.error_count += 1  
                    if self.error_count >= self.max_retries:  
                        return "no answer"  
                    

    async def _handle_triage(self, node: MemoryTreeNode, query: str) -> MemoryTreeNode:
        summaries = {i: child.content for i, child in enumerate(node.children)}
        response = await self._call_model(
            prompt=self.triage_prompt,
            inputs={"summaries": "\n".join([f"Summary {i}: {s}" for i, s in summaries.items()]), "query": query},
            model_type=self.model_type
        )
        # 解析响应
        # action = int(response.split("Action:")[-1].strip())
        # self.working_memory.append(node.children[action].content[:200])  # 保存工作记忆
        # return node.children[action]
        
        return response

    async def _handle_leaf(self, node: MemoryTreeNode, query: str) -> str:
        response = await self._call_model(
            prompt=self.leaf_prompt,
            inputs={
                "working_memory": "\n".join(self.working_memory[-3:]),  # 保留最近3条记忆
                "text": node.content,
                "query": query
            },
            model_type=self.model_type
        )
        return response
    
    
    def _parse_triage(self, response:str)->Tuple[str,str]:
        """  
        解析非叶节点响应，返回(reasoning, action)  
        符合论文中"Action must be a valid child index"的要求  
        
        return (reasoning:str, action:int)
        """  
        # 匹配格式化的响应块  
        match = re.search(  
            r"Reasoning:\s*(?P<reasoning>.+?)\s*Action:\s*(?P<action>\d+)\s*",  
            response,  
            re.DOTALL  
        )  
        
        if not match:  
            raise ParseError(f"Invalid triage response format:\n{response}")  
            
        action_str = match.group("action").strip()  
        if not action_str.isdigit():  
            raise ParseError(f"Non-numeric action: {action_str}")  
            
        return (  
            match.group("reasoning").strip(),  
            int(action_str)  
        )    
    
    def _parse_leaf(self, response:str):
        """  
        解析叶节点响应，返回(reasoning, action, answer)  
        """  
        # 匹配格式化的响应块  
        match = re.search(  
            r"\s*Reasoning:\s*(?P<reasoning>.+?)\s*Action:\s*(?P<action>-1|-2)\s*(Answer:\s*(?P<answer>.+?))?\s*",  
            response,  
            re.DOTALL  
        )  
        
        if not match:  
            raise ParseError(f"Invalid leaf response format:\n{response}")  
            
        action = int(match.group("action"))  
        answer = match.group("answer").strip() if match.group("answer") else None  
        
        # 验证逻辑  
        if action == -2 and not answer:  
            raise ParseError("Action -2 requires an answer")  
        if action == -1 and answer:  
            raise ParseError("Action -1 cannot have an answer")  
            
        return (  
            match.group("reasoning").strip(),  
            action,  
            answer  
        )  
    
    
    def _save_work_memory(self, node:MemoryTreeNode, action):
        self.working_memory.append(node.children[action].content[:200])
        print("===================================")
        print("Save a child node to the working memory, node info:")
        node.children[action].print_node_info()
        print("=====================================")
        
    
    def _find_unvisited_child(self, nodes):  
        """论文3.2节：寻找未访问的子节点"""  
        for node in nodes:  
            if node not in self.visited:  
                return node  
        return None  

    # async def _call_model(self, prompt: ChatPromptTemplate, inputs: Dict, model_type: Literal["api", "huggingface"]) -> str:
    #     if model_type == "api":
    #         client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
    #         response = client.chat.completions.create(
    #             model="glm-4",
    #             messages=[{"role": "user", "content": prompt.format(**inputs)}]
    #         )
    #         return response.choices[0].message.content
    #     elif model_type == "huggingface":
    #         llm = HuggingFacePipeline.from_model_id(
    #             model_id=MODEL_PATH,
    #             task="text-generation",
    #             max_new_tokens=1000,
    #         )
    #         chain = LLMChain(llm=llm, prompt=prompt)
    #         return await chain.arun(inputs)
    #     else:
    #         raise ValueError("Invalid model_type, please choose either 'api' or 'huggingface'")






class ChatPDFForMemWalker(ChatPDF):
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.all_chunks:List[Document] = []  # 存储分块结果  
        
    def ingest_all(self, pdf_folder_path: str):  
        """重写父类方法以存储分块结果"""  
        self.all_chunks = []  
        for file_name in os.listdir(pdf_folder_path):  
            if file_name.endswith(".pdf"):  
                file_path = os.path.join(pdf_folder_path, file_name)  
                docs = PyPDFLoader(file_path=file_path).load()  
                chunks = self.text_spliter.split_documents(docs)  
                chunks = filter_complex_metadata(chunks)  
                self.all_chunks.extend(chunks)  
        
        # 初始化向量存储（保持原有功能）  
        self.vector_store = Chroma.from_documents(  
            documents=self.all_chunks,   
            # embedding=FastEmbedEmbeddings(model_name = "BAAI/bge-small-en-v1.5")   # 注意，这玩意儿不支持本地路径
            embedding=HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                model_kwargs={'device': 'cpu'},  # 或 'cpu'  
                encode_kwargs={'normalize_embeddings': True}  
                
                )
        )  
        self.retriever = self.vector_store.as_retriever(  
            search_type="similarity_score_threshold",  
            search_kwargs={"k": 5, "score_threshold": 0.5}  
        )  
    
    def get_memwalker_chunks(self) -> List[Document]:  
        """获取适用于MEMWALKER的分块文档"""  
        return [  
            Document(  
                page_content=chunk.page_content,  
                metadata={"source": chunk.metadata.get("source", "")}  
            ) for chunk in self.all_chunks  
        ]  








async def main():
    # 初始化构建器
    builder = MemoryTreeBuilder()
    
    # 构建内存树
    # with open("long_text.txt") as f:
    #     text = f.read()
    
    pdf_reader = ChatPDFForMemWalker()
    pdf_reader.ingest_all(pdf_folder_path="src/agents/travel_knowledge/tour_pdfs")
    all_chunks  = pdf_reader.get_memwalker_chunks()
    root = await builder.build_tree(all_chunks, model_type="api")
    
    builder.print_memory_tree(root)
    
 
    navigator = Navigator(model_type="api")
    answer = await navigator.navigate(root, "如果我只有3天假期，我应该在广州怎么玩比较好？")
    print("Final Answer:", answer)
    
    
    '''
    async关键字用于定义异步函数（也称为协程）。在您提供的代码中，async def build_tree定义了一个异步方法。它的主要作用包括：

        非阻塞执行：异步函数允许程序在等待I/O操作（如网络请求、文件读写等）时，可以暂停当前任务并执行其他任务，而不是阻塞整个程序。

        提高效率：在处理大量I/O密集型任务时，异步编程可以显著提高程序的执行效率，因为它可以同时处理多个任务，而不是顺序执行。

        与await配合使用：在异步函数内部，可以使用await关键字来等待其他异步操作完成。例如在build_tree方法中，await self._create_leaf_nodes和await self._summarize_nodes就是等待这些异步操作完成。
    
    '''
    
    



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())