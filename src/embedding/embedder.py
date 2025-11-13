from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union


class TextEmbedder:
    def __init__(self, model_name_or_path: str) -> None:
        self.model = SentenceTransformer(model_name_or_path)

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()





class LangchainCompatibleEmbedder(TextEmbedder):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__(model_name_or_path)
