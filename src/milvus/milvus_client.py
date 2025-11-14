from langchain_milvus import Milvus
from pymilvus import connections, utility, db
import os, sys
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.milvus_config import MilvusConfig


class MilvusClient:
    def __init__(self, milvus_config: MilvusConfig = MilvusConfig()):
        self.milvus_config = milvus_config
        self.milvus = Milvus(
            collection_name=milvus_config.collection_name,
            embedding_function=milvus_config.embedding_function,
            connection_args=milvus_config.connection_args,
        )


    def check_if_db_exists(self):
        """检查数据库是否存在"""
        return self.milvus._collection_name in self.milvus._client.list_collections()


    def check_if_collection_exists(self):
        """检查集合是否存在"""
        return self.milvus._collection_name in self.milvus._client.list_collections()


    def create_db(self):
        """创建数据库"""
        if not self.check_if_db_exists():
            self.milvus._client.create_database(self.milvus._collection_name)


    def create_collection(self, collection_name: str = None):
        """创建集合"""
        if not self.check_if_collection_exists():
            self.milvus._client.create_collection(collection_name or self.milvus._collection_name)
