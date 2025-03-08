from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata


from langchain_community.llms.tongyi import Tongyi
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


from typing import Literal, List, Dict, Tuple

import os


DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")


class ChatPDF:

    
    def __init__(
        self, 
        prompt:ChatPromptTemplate = None,
        model_type:Literal["ollama","huggingface","tongyi","openai"]= "ollama"
        ): 
        
        if model_type == 'ollama':
            self.model = ChatOllama(model="mistral")
        elif model_type == 'huggingface':
            model_id = ""
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            hf_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
            pipe = pipeline(
                "text-generation",
                model=hf_model,
                tokenizer=tokenizer,
                max_new_tokens=200,
            )
            
            self.model = HuggingFacePipeline(pipeline=pipe)
        elif model_type == 'tongyi':
            # 运行之前： 
            # 1. 安装dashscope python sdk: pip install dashscope
            self.model = Tongyi(dashscope_api_key=DASHSCOPE_API_KEY)
        elif model_type == 'openai':
            pass
        else:
            raise ValueError("model_type must be one of 'ollama', 'huggingface', 'tongyi', 'openai'")
        
        
        self.text_spliter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 100)
        
        
        
        self.prompt = PromptTemplate.from_template(
                        """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        ) if prompt is None else prompt
        
        self.vector_store = None
        self.retriever = None
        self.chain = None
        
    def ingest_all(self, pdf_folder_path: str):
        assert self.vector_store is None, "Please clear the vector store first."
        
        all_chunks = []
        for file_name in os.listdir(pdf_folder_path):
            docs = PyPDFLoader(file_path=file_name).load()
            chunks = self.text_spliter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
            all_chunks.extend(chunks)
            
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = self.vector_store.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {
                "k": 5,
                "score_threshold": 0.5
            }
        )
        
        self.chain = ({"context":self.retriever, "question":RunnablePassthrough()}  # RunnablePassThrough 的本质， 是对用户输入的query做了恒等映射
                      | self.prompt
                      | self.model
                      | StrOutputParser()
                      )
        
        
    
    
    def ingest(self, pdf_file_path: str):
        
        assert self.vector_store is None, "Please clear the vector store first."
        
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_spliter.split_documents(docs)
        
        chunks = filter_complex_metadata(chunks)
        
        
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = self.vector_store.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {
                "k": 5,
                "score_threshold": 0.5
            }
        )
        
        self.chain = ({"context":self.retriever, "question":RunnablePassthrough()}  # RunnablePassThrough 的本质， 是对用户输入的query做了恒等映射
                      | self.prompt
                      | self.model
                      | StrOutputParser()
                      )
        
    
    
    def ask(self, query:str):
        if not self.chain:
            return "Please, add a PDF document first."
        
        
        return self.chain.invoke(query)
    
    
    
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        
        
        




if __name__ == "__main__":
    agent = ChatPDF(model_type="tongyi")
    
    agent.ingest("travel_knowledge\\tour_pdfs")

    print(agent.ask("帮我规划一个去上海的3天旅行计划。"))



