from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Optional, Union


class TextEmbedder:
    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        """
        文本嵌入器
        
        Args:
            model_name_or_path: 模型名称或路径
            device: 设备类型 ("cpu", "cuda", "mps" for Apple Silicon)
        """
        self.device = device
        self.model = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            model_kwargs={
                'device': device,
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True  # 改为 True，语义搜索通常使用归一化向量
            }
        )

    def embed_query(self, text: str) -> List[float]:
        """单个文本嵌入"""
        if not text.strip():
            return []
        return self.model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文本嵌入"""
        if not texts or all(not text.strip() for text in texts):
            return []
        # 过滤空文本
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return []
        return self.model.embed_documents(valid_texts)
        




if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    embedder = TextEmbedder(os.getenv("EMBEDDING_MODEL_PATH"))
    print(embedder.embed_query("你好"))
    print(embedder.embed_documents(["你好", "你好吗"]))

