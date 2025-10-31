import os
from typing import Optional, List
from dataclasses import dataclass, field, fields


from enum import Enum


class RAGType(str, Enum):
    MEM_WALKER = "MEM_WALKER"
    RAG = "RAG"
    SELF_RAG = "SELF_RAG"
    CORRECTIVE_RAG = "CORRECTIVE_RAG" 



@dataclass
class RAGConfig:
    pass
