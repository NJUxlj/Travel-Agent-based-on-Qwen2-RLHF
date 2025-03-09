
from typing import List, Optional, Dict, Literal, Tuple, Any
from langchain_core.documents import Document  
from langchain.text_splitter import CharacterTextSplitter 
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.llm import LLMChain



from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings  
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.vectorstores.utils import filter_complex_metadata





from langchain_core.runnables import RunnableLambda, RunnableParallel  
 
from zhipuai import ZhipuAI  
import os  

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline

from src.configs.config import MODEL_PATH,EMBEDDING_MODEL_PATH_BPE, SFT_MODEL_PATH
from src.agents.chat_pdf import ChatPDF

import asyncio

os.environ["ZHIPU_API_KEY"] = "your_api_key"  




class SelfRAGBase:
    
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
            tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code = True)
            model = AutoModelForSeq2SeqLM
            pipe = pipeline(
                task="text-generation",
                model = model,
                tokenizer=tokenizer,
                max_new_tokens = 200,
            )
            model = HuggingFacePipeline(pipeline=pipe)
            chain = LLMChain(llm=model, prompt=prompt)
            return await chain.arun(inputs)
        else:
            raise ValueError("Invalid model_type, please choose either 'api' or 'huggingface'")
        
        
    def _init_vector_store(self):  
        """初始化本地PDF文档库"""  
        if not self._vector_store:  
            # 加载本地PDF文档  
            loader = PyPDFLoader("knowledge_base.pdf")  
            documents = loader.load()  
            
            # 文档分块  
            text_splitter = RecursiveCharacterTextSplitter(  
                chunk_size=500,  
                chunk_overlap=50  
            )  
            splits = text_splitter.split_documents(documents)  
            
            # 创建向量存储  
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh")  
            self._vector_store = Chroma.from_documents(  
                documents=splits,  
                embedding=embeddings  
            )  





class SelfRAG(SelfRAGBase):  
    def __init__(self, model_type="api"):  
        self.client = ZhipuAI()  
        self.model_type = model_type  
        self.retrieve_threshold = 0.2  
        self.beam_width = 2  
        
    def retrieve(self, query: str, k: int=5) -> List[str]:  
        """检索模块"""  
        response = self._call_model()

    def generate(self, prompt: str, context: Optional[str]=None) -> str:  
        """生成模块"""  
        full_prompt = f'''上下文：{context}\n\n生成：{prompt}''' if context else prompt  
        if self.model_type == "api":  
            response = self.client.chat.completions.create(  
                model="glm-4",  
                messages=[{"role": "user", "content": full_prompt}]  
            )  
            return response.choices[0].message.content  
        else:  
            from transformers import pipeline  
            generator = pipeline("text-generation", model="Qwen/Qwen1.5-7B")  
            return generator(full_prompt)[0]['generated_text']  

    def critique(self, text: str, aspect: str) -> float:  
        """评论模块"""  
        aspects = {  
            "ISREL": "评估以下内容的相关性（Relevant/Irrelevant）：",  
            "ISSUP": "评估支持度（Fully/Partially/No support）：",  
            "ISUSE": "评估整体效用（1-5分）："  
        }  
        prompt = f"{aspects[aspect]}\n{text}"  
        
        if self.model_type == "api":  
            response = self.client.chat.completions.create(  
                model="glm-4",  
                messages=[{"role": "user", "content": prompt}]  
            )  
            return self._parse_critique(response.choices[0].message.content, aspect)  
        else:  
            from transformers import pipeline  
            classifier = pipeline("text-classification", model="Qwen/Qwen1.5-7B")  
            return classifier(prompt)[0]['score']  

    def _parse_critique(self, response: str, aspect: str) -> float:  
        """解析评论结果"""  
        if aspect == "ISUSE":  
            return int(response.strip()) / 5  
        return 1.0 if "Relevant" in response or "Fully" in response else 0.5  

    def build_chain(self):  
        """构建Self-RAG链"""  
        return (  
            RunnableParallel({  
                "input": lambda x: x,  
                "retrieve_decision": self._retrieve_decision_chain()  
            })  
            .pipe(self._generate_with_retrieval)  
            .pipe(self._critique_and_select)  
        )  

    def _retrieve_decision_chain(self):  
        """检索决策链"""  
        return (  
            RunnableLambda(lambda x: x)  
            | StrOutputParser()  
            | RunnableLambda(self._should_retrieve)  
        )  

    def _should_retrieve(self, text: str) -> dict:  
        """判断是否需要检索"""  
        score = self.critique(text, "ISREL")  
        return {"retrieve": score > self.retrieve_threshold, "text": text}  

    def _generate_with_retrieval(self, data: dict) -> dict:  
        """带检索的生成"""  
        if data["retrieve_decision"]["retrieve"]:  
            contexts = self.retrieve(data["input"], k=3)  
            return {"contexts": contexts, "input": data["input"]}  
        return {"contexts": [], "input": data["input"]}  

    def _critique_and_select(self, data: dict) -> str:  
        """评估并选择最佳结果"""  
        candidates = []  
        for context in data.get("contexts", [])[:self.beam_width]:  
            full_prompt = f"上下文：{context}\n问题：{data['input']}"  
            generation = self.generate(full_prompt)  
            
            scores = {  
                "ISREL": self.critique(f"{data['input']}\n{generation}", "ISREL"),  
                "ISSUP": self.critique(f"{context}\n{generation}", "ISSUP"),  
                "ISUSE": self.critique(generation, "ISUSE")  
            }  
            
            candidates.append({  
                "text": generation,  
                "score": 0.4*scores["ISREL"] + 0.4*scores["ISSUP"] + 0.2*scores["ISUSE"]  
            })  
        
        if not candidates:  
            return self.generate(data["input"])  
            
        return max(candidates, key=lambda x: x["score"])["text"]  



def main():

    # 使用示例  
    rag = SelfRAG(model_type="api")  # 切换为"huggingface"使用本地模型  
    chain = rag.build_chain()  

    question = "量子计算如何突破传统加密？"  
    result = chain.invoke(question)  
    print(f"最终答案：{result}")  