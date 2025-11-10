from pathlib import Path  
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pydantic import BaseModel
import os, sys
from pathlib import Path




@dataclass
class RAGConfig:
    # PDF文件夹路径
    pdf_folder_path: str = "src/agents/travel_knowledge/tour_pdfs"
    # 页面文件夹路径
    page_folder_path: str = "src/agents/travel_knowledge/tour_pages"
