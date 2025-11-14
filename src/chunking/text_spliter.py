from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional
from uuid import uuid4


class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 使用 langchain 的 RecursiveCharacterTextSplitter
        # 初始化 langchain 的递归字符文本分割器，用于将长文本按指定规则切分成小块
        self.langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,          # 每块的最大字符数
            chunk_overlap=chunk_overlap,    # 相邻块之间的重叠字符数，保持上下文连贯
            length_function=len,            # 计算长度的函数，这里使用 Python 内置的 len 计算字符数
            separators=["\n\n", "\n", " ", ""]  # 分割优先级：先按双换行，再单换行，再空格，最后按字符逐个切
        )

    def split_text(self, text: str) -> List[str]:
        """分割文本为chunks"""
        if not text.strip():
            return []
        return self.langchain_splitter.split_text(text)

    def split_documents(self, documents: List[dict]) -> List[dict]:
        """分割文档列表，每个文档包含 'page_content' 字段"""
        split_docs = []
        for doc in documents:
            if 'page_content' in doc and doc['page_content'].strip():
                chunks = self.split_text(doc['page_content'])
                for chunk in chunks:
                    split_doc = doc.copy()
                    split_doc['page_content'] = chunk
                    split_docs.append(split_doc)
        return split_docs

    def create_document_from_chunk(self, chunk: str, metadata: Optional[dict] = None) -> Document:
        """将文本块包装成 langchain Document 对象
        
        Args:
            chunk: 文本块内容
            metadata: 元数据字典，可选
            
        Returns:
            Document: langchain Document 对象
        """
        if not chunk.strip():
            raise ValueError("文本块不能为空")
            
        if metadata is None:
            metadata = {}
            
        return Document(page_content=chunk, metadata=metadata)

    def create_documents_from_chunks(self, chunks: List[str], base_metadata: Optional[dict] = None) -> List[Document]:
        """将多个文本块包装成 Document 对象列表
        
        Args:
            chunks: 文本块列表
            base_metadata: 基础元数据字典，会添加到每个Document中
                - chunk_index: 块索引，格式为 "ch_{uuid4().hex}_{i}"
                - chunk_size: 块字符数
            
        Returns:
            List[Document]: Document 对象列表
        """
        if not chunks:
            return []
            
        if base_metadata is None:
            base_metadata = {}
            
        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                # 为每个chunk添加chunk索引信息
                metadata = base_metadata.copy()
                metadata['chunk_index'] = f"ch_{uuid4().hex}_{i}"
                metadata['chunk_size'] = len(chunk)
                
                doc = self.create_document_from_chunk(chunk, metadata)
                documents.append(doc)
                
        return documents


        

    def split_text_to_documents(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """直接分割文本并返回 Document 对象列表
        
        Args:
            text: 要分割的文本
            metadata: 元数据字典，可选
            
        Returns:
            List[Document]: Document 对象列表
        """
        if not text.strip():
            return []
            
        chunks = self.split_text(text)
        return self.create_documents_from_chunks(chunks, metadata)

    